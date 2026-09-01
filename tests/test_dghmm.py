import unittest
import torch
import numpy as np
import pandas as pd
from src.models.encoder import DeepEncoder
from src.models.decoder import DeepDecoder
from src.models.hmm import DifferentiableHMM
from src.models.dghmm import DGHMM
from src.utils.flows import NormalizingFlow
from src.utils.data_loader import preprocess_mimic_demo, get_dataloader
from src.evaluation.fidelity import compute_ks_score, compute_correlation_preservation
from src.evaluation.temporal import compute_autocorrelation, compute_dtw_distance
from src.evaluation.privacy import evaluate_mia_resistance, compute_k_anonymity

class TestDGHMM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 8
        self.latent_dim = 4
        self.num_states = 3
        self.device = 'cpu'
        
        # Create random input sequence
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

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
        state_onehot[:, :, 0] = 1.0  # mock all in state 0
        
        x_recon = decoder(z, state_onehot)
        self.assertEqual(x_recon.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_hmm(self):
        hmm = DifferentiableHMM(num_states=self.num_states, latent_dim=self.latent_dim)
        z = torch.randn(self.batch_size, self.seq_len, self.latent_dim)
        
        # Test forward loss computation
        loss = hmm.forward_loss(z)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)  # scalar
        
        # Test viterbi path decoding
        states = hmm.viterbi(z)
        self.assertEqual(states.shape, (self.batch_size, self.seq_len))
        self.assertTrue(states.min() >= 0)
        self.assertTrue(states.max() < self.num_states)

    def test_dghmm_joint(self):
        model = DGHMM(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_states=self.num_states
        )
        
        # Compute joint loss
        loss, details = model.compute_loss(self.x)
        self.assertTrue(torch.is_tensor(loss))
        self.assertIn('loss_total', details)
        self.assertIn('loss_recon', details)
        self.assertIn('loss_hmm', details)
        self.assertIn('loss_kl', details)
        
        # Test backward pass (gradient flow verification)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient!")

    def test_generation(self):
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
        self.assertTrue(torch.allclose(z, reconstructed, atol=1e-4))

    def test_data_loader(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock CHARTEVENTS.csv
            data = {
                'icustay_id': [10006, 10006, 10006, 10011, 10011],
                'itemid': [220045, 220179, 220180, 220210, 220277],
                'charttime': [
                    '2164-10-23 21:00:00', '2164-10-23 22:00:00', '2164-10-23 23:00:00',
                    '2126-08-28 15:00:00', '2126-08-28 16:00:00'
                ],
                'valuenum': [85.0, 120.0, 80.0, 18.0, 98.0]
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(tmpdir, "CHARTEVENTS.csv"), index=False)
            
            sequences, feature_names = preprocess_mimic_demo(tmpdir, seq_len=4)
            
            # Expected shape: (2 stays, 4 steps, 6 features)
            self.assertEqual(sequences.shape, (2, 4, 6))
            self.assertEqual(len(feature_names), 6)
            
            loader = get_dataloader(sequences, batch_size=2, shuffle=False)
            for batch in loader:
                self.assertEqual(batch.shape, (2, 4, 6))

    def test_evaluation_metrics(self):
        real = np.random.randn(10, 5, 2)
        synth = np.random.randn(10, 5, 2)
        
        # Test fidelity
        ks = compute_ks_score(real, synth)
        corr_pres = compute_correlation_preservation(real, synth)
        self.assertTrue(0.0 <= ks <= 1.0)
        self.assertTrue(-1.0 <= corr_pres <= 1.0)
        
        # Test temporal
        acf = compute_autocorrelation(real, lag=1)
        self.assertTrue(isinstance(acf, float))
        
        # Test privacy
        real_train = real[:6]
        real_test = real[6:]
        mia = evaluate_mia_resistance(real_train, real_test, synth)
        k_anon = compute_k_anonymity(synth)
        self.assertTrue(0.0 <= mia <= 1.0)
        self.assertTrue(k_anon >= 1)

    def test_counterfactual_and_tstr(self):
        from src.evaluation.clinical_utility import evaluate_tstr
        # Test counterfactual path steering
        model = DGHMM(self.input_dim, self.latent_dim, self.num_states)
        target_states = torch.tensor([[0, 0, 1, 2, 0]], dtype=torch.long)
        counterfactual_vitals = model.generate_counterfactual(target_states)
        self.assertEqual(counterfactual_vitals.shape, (1, 5, self.input_dim))

        
        # Test TSTR utility function
        real_tr = np.random.randn(10, 8, 4) * 10 + 80
        real_te = np.random.randn(6, 8, 4) * 10 + 80
        synth = np.random.randn(12, 8, 4) * 10 + 80
        tstr_out = evaluate_tstr(real_tr, real_te, synth, horizon_hours=2)
        self.assertIn('trtr', tstr_out)
        self.assertIn('tstr', tstr_out)
        self.assertTrue(0.0 <= tstr_out['relative_utility_pct'] <= 200.0)

if __name__ == '__main__':
    unittest.main()


