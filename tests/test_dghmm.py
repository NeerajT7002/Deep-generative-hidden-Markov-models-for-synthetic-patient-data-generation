import unittest
import torch
import numpy as np
import pandas as pd
from src.models.encoder import DeepEncoder
from src.models.decoder import DeepDecoder
from src.models.hmm import DifferentiableHMM
from src.models.dghmm import DGHMM
from src.models.state_selection import evaluate_bic_aic
from src.utils.flows import NormalizingFlow
from src.utils.data_loader import preprocess_mimic_demo, get_dataloader
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import (
    compute_autocorrelation, 
    compute_dtw_distance,
    compute_skewness_error,
    compute_kurtosis_error
)
from src.evaluation.privacy import evaluate_mia_resistance, compute_min_distance, compute_k_anonymity
from src.evaluation.clinical_validity import (
    compute_clinical_rule_compliance,
    compute_physiological_bounds_adherence
)

class TestDGHMM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 8
        self.input_dim = 6
        self.latent_dim = 4
        self.num_states = 3
        self.device = 'cpu'
        
        # Create random input sequence
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        self.feature_names = ["Heart Rate", "Systolic BP", "Diastolic BP", "Respiratory Rate", "SpO2", "Temperature"]

    def test_encoder(self):
        encoder = DeepEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        z, mu, logvar = encoder(self.x)
        
        self.assertEqual(z.shape, (self.batch_size, self.seq_len, self.latent_dim))
        self.assertEqual(mu.shape, (self.batch_size, self.seq_len, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.seq_len, self.latent_dim))

    def test_decoder(self): 
        decoder = DeepDecoder(latent_dim=self.latent_dim, num_states=self.num_states, output_dim=self.input_dim)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        state_onehot = torch.zeros(self.batch_size, self.seq_len, self.num_states)
        state_onehot[:, :, 0] = 1.0  # state 0
        
        x_recon = decoder(z, state_onehot)
        self.assertEqual(x_recon.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_hmm_gmm(self):
        hmm = DifferentiableHMM(num_states=self.num_states, latent_dim=self.latent_dim, use_flows=False)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        
        loss = hmm.forward_loss(z)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        
        states = hmm.viterbi(z)
        self.assertEqual(states.shape, (self.batch_size, self.seq_len))
        self.assertTrue(states.min() >= 0)
        self.assertTrue(states.max() < self.num_states)

    def test_hmm_flows(self):
        # Test Normalizing Flow integration into HMM (Table 6)
        hmm_flow = DifferentiableHMM(num_states=self.num_states, latent_dim=self.latent_dim, use_flows=True)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        
        log_emissions = hmm_flow.compute_log_emissions(z)
        self.assertEqual(log_emissions.shape, (self.batch_size, self.seq_len, self.num_states))
        
        loss = hmm_flow.forward_loss(z)
        self.assertTrue(torch.is_tensor(loss))
        
        sample_z = hmm_flow.sample_latent_for_state(0, num_samples=5)
        self.assertEqual(sample_z.shape, (5, self.latent_dim))

    def test_dghmm_joint(self):
        model = DGHMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_states=self.num_states
        )
        
        loss, details = model.compute_loss(self.x)
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn('loss_total', details)
        self.assertIn('loss_recon', details)
        self.assertIn('loss_hmm', details)
        self.assertIn('loss_kl', details)
        
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient!")

    def test_generation_pure(self):
        model = DGHMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_states=self.num_states
        )
        
        num_gen = 5
        gen_len = 12
        synth_x, synth_states = model.generate(num_samples=num_gen, seq_len=gen_len)
        
        self.assertEqual(synth_x.shape, (num_gen, gen_len, self.input_dim))
        self.assertEqual(synth_states.shape, (num_gen, gen_len))

    def test_flows(self):
        flow = NormalizingFlow(dim=self.latent_dim, num_layers=2)
        z = torch.randn(self.batch_size, self.latent_dim)
        
        transformed, log_det = flow(z)
        self.assertEqual(transformed.shape, z.shape)
        self.assertEqual(log_det.shape, (self.batch_size,))
        
        reconstructed = flow.inverse(transformed)
        self.assertTrue(torch.allclose(z, reconstructed, atol=1e-3))
        
        log_p = flow.log_prob(z)
        self.assertEqual(log_p.shape, (self.batch_size,))

    def test_clinical_validity(self):
        # Create clinically plausible cohort
        # HR=80, SBP=120, DBP=80, RR=16, SpO2=98, Temp=98.6
        mock_cohort = np.tile([80.0, 120.0, 80.0, 16.0, 98.0, 98.6], (10, 8, 1))
        
        phys_adherence = compute_physiological_bounds_adherence(mock_cohort, self.feature_names)
        self.assertEqual(phys_adherence, 100.0)
        
        rule_compliance = compute_clinical_rule_compliance(mock_cohort, self.feature_names)
        self.assertEqual(rule_compliance, 100.0)

    def test_evaluation_metrics(self):
        real = np.random.randn(10, 5, 2) * 5 + 50
        synth = np.random.randn(10, 5, 2) * 5 + 50
        
        ks = compute_ks_score(real, synth)
        corr_pres = compute_correlation_preservation(real, synth)
        self.assertTrue(0.0 <= ks <= 1.0)
        self.assertTrue(-1.0 <= corr_pres <= 1.0)
        
        acf1 = compute_autocorrelation(real, lag=1)
        acf10 = compute_autocorrelation(real, lag=4)
        dtw = compute_dtw_distance(real, synth)
        skew_err = compute_skewness_error(real, synth)
        kurt_err = compute_kurtosis_error(real, synth)
        
        self.assertTrue(isinstance(acf1, float))
        self.assertTrue(isinstance(dtw, float))
        self.assertTrue(skew_err >= 0.0)
        self.assertTrue(kurt_err >= 0.0)
        
        mia = evaluate_mia_resistance(real[:6], real[6:], synth)
        min_d = compute_min_distance(real, synth)
        k_anon = compute_k_anonymity(synth)
        self.assertTrue(0.0 <= mia <= 1.0)
        self.assertTrue(min_d >= 0.0)
        self.assertTrue(k_anon >= 1.0)

    def test_counterfactual_and_tstr(self):
        from src.evaluation.clinical_utility import evaluate_tstr
        model = DGHMM(self.input_dim, self.latent_dim, self.num_states)
        target_states = torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.long)
        counterfactual_vitals = model.generate_counterfactual(target_states)
        self.assertEqual(counterfactual_vitals.shape, (1, 5, self.input_dim))
        
        real_tr = np.random.randn(10, 8, 4) * 10 + 80
        real_te = np.random.randn(6, 8, 4) * 10 + 80
        synth = np.random.randn(12, 8, 4) * 10 + 80
        tstr_out = evaluate_tstr(real_tr, real_te, synth, horizon_hours=2)
        self.assertIn('trtr', tstr_out)
        self.assertIn('tstr', tstr_out)

    def test_state_selection(self):
        mock_data = np.random.randn(12, 6, self.input_dim)
        res = evaluate_bic_aic(mock_data, candidate_states=[2, 3], epochs=2, batch_size=4)
        self.assertIn('optimal_k_bic', res)
        self.assertIn('optimal_k_aic', res)

if __name__ == '__main__':
    unittest.main()
