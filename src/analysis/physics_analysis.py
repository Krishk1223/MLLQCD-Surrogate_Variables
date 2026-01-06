"""Bayesian fitting and PDF report generation for 2pt correlators."""

import json
import numpy as np
import gvar as gv
import corrfitter as cf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .preliminary_analysis import PreliminaryAnalysis2pt, PreliminaryMLAnalysis2pt

TIME_EXTENT = 96
MODELS = ['cnn', 'gbr', 'mlp', 'transformer']
COLOURS = {
    'truth': '#1f77b4',  # Blue
    'cnn': '#ff7f0e',    # Orange
    'gbr': '#2ca02c',    # Green
    'mlp': '#d62728',    # Red
    'transformer': '#9467bd',  # Purple
    'RidgeRegression': '#e377c2',  # Pink (placeholder)
}

# Markers for different correction types
MARKERS = {
    'bias_corrected': 'o',
    'ratio': 's',
    'boosted_ratio': '^',
}

def build_2pt_prior(estimates: Dict, nstates: int = 2, 
                    n_normal: int = None, n_oscillating: int = None,
                    dE_excited: float = 0.2, dE_width: float = None) -> gv.BufferDict:
    """Build prior dict for 2pt fit from plateau estimates.
    
    Uses log-normal priors for amplitudes and energy gaps to ensure:
    - Amplitudes are positive
    - Energy gaps are positive (E_n > E_{n-1})
    
    The energy structure is:
    - E0: ground state energy (from estimates)
    - dE[0] = E0 (the ground state energy itself, confusingly named by corrfitter)
    - dE[n] for n>0: energy GAPS between states, i.e., E_n - E_{n-1}
    
    Args:
        estimates: Dict with lnE0, lna0 etc from preliminary analysis
        nstates: Number of states for both normal and oscillating (if n_normal/n_oscillating not set)
        n_normal: Number of normal parity states (overrides nstates)
        n_oscillating: Number of oscillating parity states (overrides nstates)
        dE_excited: Central value for excited state energy gaps (default 0.2)
        dE_width: Width for log(dE) prior (default log(2) ~ 0.69)
    """
    prior = gv.BufferDict()
    
    # Set default width to log(2) for factor-of-2 uncertainty
    if dE_width is None:
        dE_width = np.log(2)  # ~0.69, means factor of 2 uncertainty
    
    # Use specific counts if provided, else use nstates for both
    n_norm = n_normal if n_normal is not None else nstates
    n_osc = n_oscillating if n_oscillating is not None else nstates
    
    # Ground state energy and amplitude from estimates
    lnE0 = estimates.get('lnE0', np.log(0.8))  # Default ~0.8 for D meson
    lna0 = estimates.get('lna0', -2.0)
    
    # Prior for excited state energy gaps: dE ~ 0.2 with factor-of-2 uncertainty
    ln_dE_excited = np.log(dE_excited)  # log(0.2) ~ -1.6
    
    # === Normal parity states ===
    # Amplitudes: ground state from estimates, excited states smaller
    log_a_means = [lna0]
    log_a_widths = [3.0]  # Wide but not too wide
    for i in range(1, n_norm):
        log_a_means.append(lna0 - 2.0 * i)  # Each excited state ~7x smaller
        log_a_widths.append(3.0)
    
    prior['log(a)'] = gv.gvar(log_a_means, log_a_widths)
    
    # Energy gaps: dE[0] is ground state E0, dE[n>0] are gaps
    log_dE_means = [lnE0]  # Ground state energy
    log_dE_widths = [1.0]  # Reasonable width for ground state
    for i in range(1, n_norm):
        log_dE_means.append(ln_dE_excited)  # Gap ~ 0.2
        log_dE_widths.append(dE_width)  # Factor of 2 uncertainty
    
    prior['log(dE)'] = gv.gvar(log_dE_means, log_dE_widths)
    
    # === Oscillating parity states ===
    lnao0 = estimates.get('lnao0', lna0 - 1.0)
    lnEo0 = estimates.get('lnEo0', lnE0)  # Similar energy for oscillating ground
    
    log_ao_means = [lnao0]
    log_ao_widths = [3.0]
    for i in range(1, n_osc):
        log_ao_means.append(lnao0 - 2.0 * i)
        log_ao_widths.append(3.0)
    
    prior['log(ao)'] = gv.gvar(log_ao_means, log_ao_widths)
    
    log_dEo_means = [lnEo0]
    log_dEo_widths = [1.0]
    for i in range(1, n_osc):
        log_dEo_means.append(ln_dE_excited)
        log_dEo_widths.append(dE_width)
    
    prior['log(dEo)'] = gv.gvar(log_dEo_means, log_dEo_widths)
    
    return prior


def build_3pt_prior(heavy_est: Dict, light_est: Dict) -> gv.BufferDict:
    """Build prior dict for simultaneous 2pt + 3pt fit."""
    prior = gv.BufferDict()
    prior['src:log(a)'] = gv.gvar(heavy_est['lna0'], heavy_est['lna0_interval'])
    prior['src:log(dE)'] = gv.gvar(heavy_est['lnE0'], heavy_est['lnE0_interval'])
    prior['src:log(ao)'] = gv.gvar(heavy_est['lna1'], heavy_est['lna1_interval'])
    prior['src:log(dEo)'] = gv.gvar(heavy_est['ln_delta_E1'], heavy_est['ln_delta_E1_interval'])
    prior['snk:log(a)'] = gv.gvar(light_est['lna0'], light_est['lna0_interval'])
    prior['snk:log(dE)'] = gv.gvar(light_est['lnE0'], light_est['lnE0_interval'])
    prior['snk:log(ao)'] = gv.gvar(light_est['lna1'], light_est['lna1_interval'])
    prior['snk:log(dEo)'] = gv.gvar(light_est['ln_delta_E1'], light_est['ln_delta_E1_interval'])
    prior['Vnn'] = gv.gvar([[1.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    prior['Vno'] = gv.gvar([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    prior['Von'] = gv.gvar([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    prior['Voo'] = gv.gvar([[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]])
    return prior

def jackknife_to_gvar(jk_samples: np.ndarray, with_cov: bool = True) -> np.ndarray:
    """Convert jackknife samples to gvar array with proper jackknife errors.
    
    The jackknife error is: error = std(samples) * sqrt(N-1)
    The jackknife covariance is: cov(samples) * (N-1)
    
    Args:
        jk_samples: (N_samples, N_timeslices) array of jackknife samples
        with_cov: If True, include full covariance matrix (more accurate but slower)
    """
    N = len(jk_samples)
    mean = np.mean(jk_samples, axis=0)
    
    if with_cov:
        # Full covariance matrix with proper jackknife scaling
        cov = np.cov(jk_samples.T) * (N - 1)
        return gv.gvar(mean, cov)
    else:
        # Just diagonal errors
        err = np.std(jk_samples, axis=0) * np.sqrt(N - 1)
        return gv.gvar(mean, err)


def apply_truth_covariance(ml_samples: np.ndarray, truth_samples: np.ndarray,
                           is_jackknife: bool = True) -> np.ndarray:
    """Apply truth covariance matrix to ML prediction mean.
    
    ML predictions have large sample-to-sample variance (model uncertainty),
    but for chi2 fitting we want statistical uncertainty. Using truth covariance
    assumes ML predictions have similar statistical precision to truth.
    
    Args:
        ml_samples: (N_ml, N_timeslices) ML prediction samples
        truth_samples: (N_truth, N_timeslices) truth samples
        is_jackknife: If True, truth_samples are jackknife resamples and we use
                      cov * (N-1) scaling. If False, they are individual configs
                      and we use cov / N for variance of mean.
    
    Returns:
        gvar array with ML mean and truth covariance (for error on the mean)
    """
    ml_mean = np.mean(ml_samples, axis=0)
    N_truth = len(truth_samples)
    
    if is_jackknife:
        # Jackknife: covariance of resampled means = sample_cov * (N-1)
        truth_cov = np.cov(truth_samples.T) * (N_truth - 1)
    else:
        # Individual configs: variance of mean = sample_cov / N
        truth_cov = np.cov(truth_samples.T) / N_truth
    
    return gv.gvar(ml_mean, truth_cov)


def fit_2pt_correlator(corr_data: np.ndarray, estimates: Dict,
                       tmin: int = None, tmax: int = None, nstates: int = 2,
                       reference_cov: np.ndarray = None) -> cf.CorrFitter:
    """Bayesian 2pt fit with oscillating terms.
    
    Uses full covariance matrix from jackknife for accurate chi2.
    Default is tmin=25, tmax=40 which gives chi2/dof ~ 1 for typical staggered data.
    """
    # Use proper jackknife covariance
    gv_data = jackknife_to_gvar(corr_data, with_cov=True)
    prior = build_2pt_prior(estimates, nstates=nstates)
    
    # Default: tmin=25 avoids early-time contamination and gives chi2/dof ~ 1
    if tmin is None:
        tmin = 25
    if tmax is None:
        tmax = 40
    # Bound sensibly
    tmin = max(1, min(tmin, 30))
    tmax = max(tmin + 5, min(tmax, 47))  # Leave room before T/2=48
    
    model = cf.Corr2(
        datatag='2pt',
        tp=TIME_EXTENT,
        tmin=tmin,
        tmax=tmax,
        a=('a', 'ao'),
        b=('a', 'ao'),
        dE=('dE', 'dEo'),
        s=(1.0, -1.0)
    )
    fitter = cf.CorrFitter(models=[model])
    return fitter.lsqfit(data={'2pt': gv_data}, prior=prior)


def auto_fit_2pt_correlator(corr_data: np.ndarray, estimates: Dict,
                            target_chi2_dof: float = 1.0,
                            truth_data: np.ndarray = None,
                            truth_is_jackknife: bool = True,
                            nstates: int = 3,
                            n_normal: int = None,
                            n_oscillating: int = None,
                            dE_excited: float = 0.2,
                            use_ml_covariance: bool = False,
                            is_jackknife: bool = False,
                            tmin_range: tuple = (5, 25),
                            tmax_options: list = None) -> cf.CorrFitter:
    """Automatically find best fit range for 2pt correlator.
    
    Scans tmin to find range with chi2/dof closest to target.
    
    Args:
        corr_data: Correlator samples (jackknife or ML predictions)
        estimates: Preliminary estimates for priors
        target_chi2_dof: Target chi2/dof value
        truth_data: If provided, use truth covariance for ML predictions
                   (gives meaningful chi2 for ML models)
        truth_is_jackknife: If True, truth_data contains jackknife resamples.
                           If False, truth_data contains individual configs.
        nstates: Number of states for both parities (default 3)
        n_normal: Number of normal parity states (overrides nstates)
        n_oscillating: Number of oscillating parity states (overrides nstates)
        dE_excited: Central value for excited state energy gaps (default 0.2)
        use_ml_covariance: If True, use ML prediction covariance (shows model differences)
        is_jackknife: If True, corr_data contains jackknife samples (use jackknife covariance)
        tmin_range: (start, end) for tmin scan
        tmax_options: list of tmax values to try
    """
    if tmax_options is None:
        tmax_options = [40, 45, 48]
    
    if use_ml_covariance or truth_data is None:
        # Use sample covariance - each model will have different chi2
        N = len(corr_data)
        ml_mean = np.mean(corr_data, axis=0)
        
        if is_jackknife:
            # For jackknife: variance = (N-1)/N * sum((x_i - xbar)^2) 
            # And for variance of mean, we don't divide by N again
            # The jackknife estimate of variance is: (N-1) * Var(jackknife samples)
            ml_cov = np.cov(corr_data.T) * (N - 1)  # Jackknife covariance
        else:
            # For regular samples: variance of mean = cov / N
            ml_cov = np.cov(corr_data.T) / N
        gv_data = gv.gvar(ml_mean, ml_cov)
    else:
        # Use truth covariance with ML mean
        gv_data = apply_truth_covariance(corr_data, truth_data, is_jackknife=truth_is_jackknife)
    
    # Build prior with excited state gaps ~ 0.2 and log(2) width
    prior = build_2pt_prior(
        estimates, nstates=nstates,
        n_normal=n_normal, n_oscillating=n_oscillating,
        dE_excited=dE_excited, dE_width=np.log(2)
    )
    
    best_fit = None
    best_score = float('inf')
    
    # Scan tmin range for good chi2/dof
    for tmin in range(tmin_range[0], tmin_range[1], 2):
        for tmax in tmax_options:
            if tmax <= tmin + 5:
                continue
            try:
                model = cf.Corr2(
                    datatag='2pt', tp=TIME_EXTENT,
                    tmin=tmin, tmax=tmax,
                    a=('a', 'ao'), b=('a', 'ao'),
                    dE=('dE', 'dEo'), s=(1.0, -1.0)
                )
                fitter = cf.CorrFitter(models=[model])
                fit = fitter.lsqfit(data={'2pt': gv_data}, prior=prior)
                chi2_dof = fit.chi2 / max(1, fit.dof)
                # Score: prefer chi2/dof close to target, with reasonable Q
                score = abs(chi2_dof - target_chi2_dof) + (0.5 if fit.Q < 0.05 else 0)
                if score < best_score:
                    best_score = score
                    best_fit = fit
            except Exception:
                continue
    
    if best_fit is None:
        # Fallback to default
        return fit_2pt_correlator(corr_data, estimates, tmin=25, tmax=40, nstates=nstates)
    return best_fit


def fit_3pt_correlator(heavy_data: np.ndarray, light_data: np.ndarray,
                       three_pt_data: Dict[int, np.ndarray],
                       heavy_est: Dict, light_est: Dict,
                       separations: List[int] = None) -> cf.CorrFitter:
    """Simultaneous 2pt + 3pt Bayesian fit."""
    separations = separations or [12, 15, 18]
    data = {
        'src': gv.dataset.avg_data(heavy_data, bstrap=False),
        'snk': gv.dataset.avg_data(light_data, bstrap=False),
    }
    for T in separations:
        data[f'3pt_T{T}'] = gv.dataset.avg_data(three_pt_data[T], bstrap=False)
    prior = build_3pt_prior(heavy_est, light_est)
    models = []
    models.append(cf.Corr2(
        datatag='src',
        tp=TIME_EXTENT,
        tmin=5, tmax=35,
        a=('src:a', 'src:ao'),
        b=('src:a', 'src:ao'),
        dE=('src:dE', 'src:dEo'),
        s=(1.0, -1.0)
    ))
    models.append(cf.Corr2(
        datatag='snk',
        tp=TIME_EXTENT,
        tmin=5, tmax=35,
        a=('snk:a', 'snk:ao'),
        b=('snk:a', 'snk:ao'),
        dE=('snk:dE', 'snk:dEo'),
        s=(1.0, -1.0)
    ))
    for T in separations:
        models.append(cf.Corr3(
            datatag=f'3pt_T{T}',
            T=T,
            tmin=3,
            tmax=T-3,
            a=('src:a', 'src:ao'),
            dEa=('src:dE', 'src:dEo'),
            sa=(1.0, -1.0),
            b=('snk:a', 'snk:ao'),
            dEb=('snk:dE', 'snk:dEo'),
            sb=(1.0, -1.0),
            Vnn='Vnn', Vno='Vno', Von='Von', Voo='Voo'
        ))
    fitter = cf.CorrFitter(models=models)
    return fitter.lsqfit(data=data, prior=prior)

def extract_fit_parameters(fit) -> Dict:
    """Extract E0, a0, dE1, a1 from fit result."""
    p = fit.p
    # Ground state energy (first element of dE array)
    dE = p.get('log(dE)', gv.gvar([0], [1]))
    E0 = gv.exp(dE[0]) if hasattr(dE, '__len__') else gv.exp(dE)
    
    # Ground state amplitude
    a = p.get('log(a)', gv.gvar([0], [1]))
    a0 = gv.exp(a[0]) if hasattr(a, '__len__') else gv.exp(a)
    
    # First excited state (second element if exists)
    dE1 = gv.exp(dE[1]) if hasattr(dE, '__len__') and len(dE) > 1 else gv.gvar(0, 1)
    a1 = gv.exp(a[1]) if hasattr(a, '__len__') and len(a) > 1 else gv.gvar(0, 1)
    
    return {
        'E0': E0,
        'a0': a0,
        'dE1': dE1,
        'a1': a1,
        'chi2_dof': fit.chi2 / max(1, fit.dof),
        'Q': fit.Q
    }

class TwoPointReport:
    """PDF report comparing truth vs ML model fits."""
    
    def __init__(self, experiment: str, project_root: Path = None):
        self.experiment = experiment
        self.root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.results = self.root / "results" / experiment
        self.data = self.root / "data"
        self.correlators = {}
        self.estimates = {}
        self.params = {}
        self.fits = {}  # Store fit objects for later access
    
    def models_ready(self) -> bool:
        missing = [m for m in MODELS if not (self.results / m / "bias_corrected" / "correlator_predictions.npy").exists()]
        if missing:
            print(f"Missing: {missing}")
        return not missing
    
    def load_truth(self) -> np.ndarray:
        exp_path = self.data / "experiments" / self.experiment
        meta_file = exp_path / "metadata.json"
        if meta_file.exists():
            with open(meta_file) as f:
                target = Path(json.load(f).get('target filename', '')).stem
            jk_file = self.data / "processed" / "jackknife_samples" / f"{target}_jackknife_samples.npy"
            if jk_file.exists():
                return np.load(jk_file)
        return np.load(exp_path / "test_data_y.npy")
    
    def load_test_targets(self) -> np.ndarray:
        """Load the test set targets (what ML was trained to predict)."""
        exp_path = self.data / "experiments" / self.experiment
        return np.load(exp_path / "test_data_y.npy")
    
    def load_ml_predictions(self, model: str, correction: str) -> Optional[np.ndarray]:
        paths = {
            'raw': self.results / model / "correlator_predictions.npy",
            'bias_corrected': self.results / model / "bias_corrected" / "correlator_predictions.npy",
            'ratio': self.results / model / "ratio_predictions" / "correlator_predictions.npy",
            'boosted_ratio': self.results / model / "boosted_ratio_predictions" / "correlator_predictions.npy",
        }
        path = paths.get(correction)
        return np.load(path) if path and path.exists() else None
    
    def apply_posthoc_bias_correction(self, ml_preds: np.ndarray, truth: np.ndarray,
                                       test_targets: np.ndarray = None) -> np.ndarray:
        """
        Apply post-hoc bias correction to ML predictions.
        
        The ML model was trained to predict test_targets. If we want to compare
        against a different truth (e.g., jackknife samples from all data), we need
        to correct for the mean difference.
        
        Correction: shift ML predictions by (truth_mean - ml_mean)
        This makes the ML mean match the truth mean exactly.
        
        If test_targets is provided, we can do a two-step correction:
        1. First match ML to test_targets (training target)
        2. Then shift from test_targets to truth
        """
        ml_mean = np.mean(ml_preds, axis=0)
        truth_mean = np.mean(truth, axis=0)
        
        # Simple approach: shift to match truth mean directly
        bias = ml_mean - truth_mean  # Shape: (T,)
        corrected = ml_preds - bias  # Broadcast over samples
        
        return corrected
    
    def compute_preliminary_estimates(self, data: np.ndarray) -> Dict:
        """Chi2-minimised plateau estimates for priors. Handles staggered oscillations.
        
        For staggered fermions, we use even timeslices only (same parity) to get
        a clean effective mass without oscillation contamination.
        """
        mean = np.mean(data, axis=0)
        err = np.std(data, axis=0) / np.sqrt(len(data))
        n = len(mean) - 2  # Need t+2
        
        # For staggered fermions: use ratio of same-parity timeslices C(t)/C(t+2)
        # This removes the oscillating contribution
        # m_eff = 0.5 * acosh((C(t-2) + C(t+2)) / (2*C(t))) is more stable
        # But simpler: m_eff = 0.5 * ln(|C(t)| / |C(t+2)|) for even t
        
        m_eff = []
        m_err = []
        times = []
        
        # Use EVEN timeslices only (same parity, no oscillation)
        for t in range(4, min(30, n), 2):
            if np.abs(mean[t]) > 1e-15 and np.abs(mean[t+2]) > 1e-15:
                # Effective mass from same-parity ratio
                m = 0.5 * np.log(np.abs(mean[t]) / np.abs(mean[t+2]))
                # Error propagation
                rel_err_t = np.abs(err[t] / mean[t]) if np.abs(mean[t]) > 1e-15 else 1.0
                rel_err_t2 = np.abs(err[t+2] / mean[t+2]) if np.abs(mean[t+2]) > 1e-15 else 1.0
                me = 0.5 * np.sqrt(rel_err_t**2 + rel_err_t2**2)
                
                if np.isfinite(m) and np.isfinite(me) and me > 0 and me < 1.0:
                    m_eff.append(m)
                    m_err.append(me)
                    times.append(t)
        
        if len(m_eff) < 3:
            return self.default_estimates()
        
        m_eff = np.array(m_eff)
        m_err = np.array(m_err)
        times = np.array(times)
        
        # Find plateau with minimal chi2
        best_chi2, best_i = np.inf, 0
        window = min(3, len(m_eff) - 1)
        for i in range(len(m_eff) - window):
            w = m_eff[i:i+window+1]
            e = m_err[i:i+window+1]
            wt = 1 / e**2
            wm = np.sum(wt * w) / np.sum(wt)
            chi2 = np.sum(((w - wm) / e)**2) / max(1, window)
            if chi2 < best_chi2:
                best_chi2, best_i = chi2, i
        
        # Get plateau energy
        pv = m_eff[best_i:best_i+window+1]
        pe = m_err[best_i:best_i+window+1]
        pt = times[best_i:best_i+window+1]
        wt = 1 / pe**2
        E0 = np.sum(wt * pv) / np.sum(wt)
        
        # Estimate amplitudes from early timeslices
        # For staggered: C(t) ~ a_n^2 * exp(-E_n*t) + (-1)^t * a_o^2 * exp(-E_o*t)
        # At even t: C(t) ~ a_n^2 * exp(-E_n*t) + a_o^2 * exp(-E_o*t)
        # At odd t: C(t) ~ a_n^2 * exp(-E_n*t) - a_o^2 * exp(-E_o*t)
        
        # Use t=0 (even) for total amplitude
        C0 = np.abs(mean[0]) if len(mean) > 0 else 0.05
        a_total = np.sqrt(C0)
        
        # Estimate oscillating amplitude from ratio of even/odd
        C1 = np.abs(mean[1]) if len(mean) > 1 else C0 * 0.1
        a_osc = np.sqrt(np.abs(C0 * np.exp(-E0) - C1) / 2) if C0 * np.exp(-E0) > C1 else a_total * 0.5
        
        # Normal parity amplitude
        a0 = max(a_total * 0.7, 0.001)
        ao0 = max(a_osc, 0.001)
        
        E0 = max(E0, 0.01)
        
        return {
            'E0': E0,
            'lnE0': np.log(E0),
            'lnE0_interval': 1.0,
            'ln_delta_E1': np.log(0.3),
            'ln_delta_E1_interval': 2.0,
            'a0': a0,
            'lna0': np.log(a0),
            'lna0_interval': 2.0,
            'lna1': np.log(max(a0 * 0.1, 0.001)),
            'lna1_interval': 3.0,
            'lnao0': np.log(ao0),
            'lnEo0': np.log(E0),  # Similar energy for oscillating
            't_min': int(pt[0]) if len(pt) > 0 else 4,
            't_max': int(pt[-1]) if len(pt) > 0 else 20
        }
    
    def default_estimates(self) -> Dict:
        return {
            'E0': 0.1,
            'lnE0': np.log(0.1),
            'lnE0_interval': 0.5,
            'ln_delta_E1': np.log(0.2),
            'ln_delta_E1_interval': np.log(3),
            'a0': 10.0,
            'lna0': np.log(10.0),
            'lna0_interval': np.log(5),
            'lna1': np.log(5.0),
            'lna1_interval': np.log(10),
            't_min': 7,
            't_max': 25
        }
    
    def compute_effective_mass(self, data: np.ndarray, staggered: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute effective mass from correlator data.
        
        Args:
            data: Correlator samples (N_samples, T)
            staggered: If True, use t+2 ratio for staggered fermions (avoids sign oscillation)
                      If False, use t+1 ratio (standard)
        
        Returns:
            t, m_eff, m_err arrays
        """
        mean = np.mean(data, axis=0)
        err = np.std(data, axis=0) / np.sqrt(len(data))
        
        if staggered:
            # For staggered fermions: use t+2 to compare same-parity timeslices
            # m_eff = 0.5 * ln(|C(t)| / |C(t+2)|)
            n = len(mean) - 2
            step = 2
            scale = 0.5  # Because we're comparing t and t+2
        else:
            # Standard: m_eff = ln(C(t) / C(t+1))
            n = len(mean) - 1
            step = 1
            scale = 1.0
        
        t_vals = []
        m_vals = []
        me_vals = []
        
        for t in range(n):
            c_t = mean[t]
            c_t_next = mean[t + step]
            
            # For staggered, use absolute values since signs alternate
            if staggered:
                c_t = np.abs(c_t)
                c_t_next = np.abs(c_t_next)
            
            if c_t > 1e-20 and c_t_next > 1e-20:
                m = scale * np.log(c_t / c_t_next)
                # Error propagation
                rel_err_t = np.abs(err[t] / mean[t]) if np.abs(mean[t]) > 1e-20 else 1.0
                rel_err_next = np.abs(err[t + step] / mean[t + step]) if np.abs(mean[t + step]) > 1e-20 else 1.0
                me = scale * np.sqrt(rel_err_t**2 + rel_err_next**2)
                
                if np.isfinite(m) and np.isfinite(me) and m > 0:
                    t_vals.append(t)
                    m_vals.append(m)
                    me_vals.append(me)
        
        return np.array(t_vals), np.array(m_vals), np.array(me_vals)
    
    def compute_noise_to_signal(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.arange(len(mean)), np.abs(std / mean)
    
    def run_fits(self, verbose: bool = True):
        """Run physics fits on truth and ML predictions.
        
        Uses ML sample covariance for fitting, which allows comparing different
        models based on their prediction quality and variance structure.
        
        Args:
            verbose: Print progress and results
        """
        if verbose:
            print("Fitting truth...")
        
        truth = self.load_truth()
        self.correlators['truth'] = truth
        self.estimates['truth'] = self.compute_preliminary_estimates(truth)
        
        try:
            # Truth uses jackknife covariance
            # Use 3 normal + 2 oscillating states with dE ~ 0.2
            fit = auto_fit_2pt_correlator(
                truth, self.estimates['truth'],
                use_ml_covariance=True,
                is_jackknife=True,
                nstates=3,
                n_normal=3, n_oscillating=2,
                dE_excited=0.2,
                tmin_range=(8, 28),
                tmax_options=[40, 45, 48]
            )
            self.fits['truth'] = fit
            self.params['truth'] = extract_fit_parameters(fit)
            if verbose:
                print(f"  Truth: chi2/dof={fit.chi2/fit.dof:.2f}, Q={fit.Q:.3f}")
        except Exception as e:
            if verbose:
                print(f"Truth fit failed: {e}")
            self.params['truth'] = None
        
        for model in MODELS:
            if verbose:
                print(f"Fitting {model}...")
            
            self.correlators[model] = {}
            self.estimates[model] = {}
            self.params[model] = {}
            
            # Only fit bias_corrected, ratio, boosted_ratio
            corrections = ['bias_corrected', 'ratio', 'boosted_ratio']
            
            for correction in corrections:
                data = self.load_ml_predictions(model, correction)
                
                if data is None:
                    continue
                
                self.correlators[model][correction] = data
                self.estimates[model][correction] = self.compute_preliminary_estimates(data)
                
                try:
                    # Use ML sample covariance - shows how each model differs
                    # Use 3 normal + 2 oscillating states with dE ~ 0.2
                    fit = auto_fit_2pt_correlator(
                        data, self.estimates[model][correction],
                        use_ml_covariance=True,
                        nstates=3,
                        n_normal=3, n_oscillating=2,
                        dE_excited=0.2,
                        tmin_range=(8, 28),
                        tmax_options=[40, 45, 48]
                    )
                    self.fits[f'{model}_{correction}'] = fit
                    self.params[model][correction] = extract_fit_parameters(fit)
                    if verbose:
                        print(f"  {model} {correction}: chi2/dof={fit.chi2/fit.dof:.2f}, Q={fit.Q:.3f}")
                except Exception as e:
                    if verbose:
                        print(f"  {model} {correction} fit failed: {e}")
    
    def plot_log_correlator(self, ax, sources: List[str], clean: bool = False, show_errors: bool = False):
        for src in sources:
            if src == 'truth':
                data = self.correlators.get('truth')
            else:
                data = self.correlators.get(src, {}).get('bias_corrected')
            
            if data is None:
                continue
            
            mean = np.mean(data, axis=0)
            err = np.std(data, axis=0) / np.sqrt(len(data))
            t = np.arange(len(mean))
            
            if clean:
                mask = t % 2 == 1
                t, mean, err = t[mask], mean[mask], err[mask]
            
            if show_errors:
                ax.errorbar(t, np.log(np.abs(mean)), yerr=err/np.abs(mean), 
                           fmt='o-', color=COLOURS.get(src, 'gray'), 
                           label=src.upper() if src != 'truth' else 'Truth', ms=3, capsize=2)
            else:
                ax.plot(t, np.log(np.abs(mean)), 'o-', color=COLOURS.get(src, 'gray'), 
                       label=src.upper() if src != 'truth' else 'Truth', ms=3)
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('ln|C(t)|', fontsize=10)
        ax.legend(fontsize=9)
    
    def plot_ml_comparison(self, ax, model: str, clean: bool = False):
        plot_data = [
            (self.correlators.get('truth'), 'Truth', 'black'),
            (self.correlators.get(model, {}).get('bias_corrected'), f'{model} (bias)', 'blue'),
            (self.correlators.get(model, {}).get('raw'), f'{model} (raw)', 'red'),
        ]
        
        for data, label, colour in plot_data:
            if data is None:
                continue
            
            mean = np.mean(data, axis=0)
            err = np.std(data, axis=0) / np.sqrt(len(data))
            t = np.arange(len(mean))
            
            if clean:
                mask = t % 2 == 1
                t, mean, err = t[mask], mean[mask], err[mask]
            
            ax.errorbar(t, np.log(np.abs(mean)), yerr=err/np.abs(mean),
                       fmt='o-', color=colour, label=label, 
                       ms=3, capsize=2, alpha=0.8)
        
        ax.set_xlabel('t')
        ax.set_ylabel('ln|C(t)|')
        ax.legend()
        ax.set_title(f'{model.upper()}: Truth vs ML')
    
    def plot_effective_mass(self, ax, sources: List[str], clean: bool = False):
        for src in sources:
            if src == 'truth':
                data = self.correlators.get('truth')
            else:
                data = self.correlators.get(src, {}).get('bias_corrected')
            
            if data is None:
                continue
            
            t, m, me = self.compute_effective_mass(data, clean)
            ax.errorbar(t, m, yerr=me, fmt='o-', 
                       color=COLOURS.get(src, 'gray'), 
                       label=src, ms=3, capsize=2)
        if self.params.get('truth'):
            E0 = self.params['truth']['E0']
            ax.axhline(gv.mean(E0), color='black', ls='--', alpha=0.5)
            ax.axhspan(gv.mean(E0) - gv.sdev(E0), 
                      gv.mean(E0) + gv.sdev(E0), 
                      alpha=0.2, color='gray')
        
        ax.set_xlabel('t')
        ax.set_ylabel('m_eff(t)')
        ax.set_xlim(0, 40)
        ax.legend()
    
    def plot_noise_to_signal(self, ax, sources: List[str]):
        for src in sources:
            if src == 'truth':
                data = self.correlators.get('truth')
            else:
                data = self.correlators.get(src, {}).get('bias_corrected')
            
            if data is None:
                continue
            
            t, nsr = self.compute_noise_to_signal(data)
            ax.semilogy(t, nsr, 'o-', color=COLOURS.get(src, 'gray'), 
                       label=src, ms=3)
        
        ax.set_xlabel('t')
        ax.set_ylabel('Noise/Signal')
        ax.legend()
    
    def plot_fit_table(self, ax):
        ax.axis('off')
        headers = ['Model', 'a0', 'E0', 'a1', 'dE1', 'χ²/dof', 'Q']
        rows = []
        if self.params.get('truth'):
            p = self.params['truth']
            rows.append([
                'Truth',
                f"{gv.mean(p['a0']):.4f}",
                f"{gv.mean(p['E0']):.4f}",
                f"{gv.mean(p['a1']):.4f}",
                f"{gv.mean(p['dE1']):.4f}",
                f"{p['chi2_dof']:.2f}",
                f"{p['Q']:.3f}"
            ])
        for model in MODELS:
            p = self.params.get(model, {}).get('bias_corrected')
            if p:
                rows.append([
                    model.upper(),
                    f"{gv.mean(p['a0']):.4f}",
                    f"{gv.mean(p['E0']):.4f}",
                    f"{gv.mean(p['a1']):.4f}",
                    f"{gv.mean(p['dE1']):.4f}",
                    f"{p['chi2_dof']:.2f}",
                    f"{p['Q']:.3f}"
                ])
        
        if rows:
            table = ax.table(cellText=rows, colLabels=headers, 
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
    
    def plot_benchmark_table(self, ax, model: str = None):
        """Benchmark table: truth vs ML+RM vs ML+bRM."""
        ax.axis('off')
        headers = ['Method', 'a0', 'E0', 'dE1', 'χ²/dof', 'Q']
        rows = []
        
        if self.params.get('truth'):
            p = self.params['truth']
            rows.append([
                'Truth',
                f"{gv.mean(p['a0']):.4f}({gv.sdev(p['a0']):.4f})",
                f"{gv.mean(p['E0']):.5f}({gv.sdev(p['E0']):.5f})",
                f"{gv.mean(p['dE1']):.4f}({gv.sdev(p['dE1']):.4f})",
                f"{p['chi2_dof']:.2f}",
                f"{p['Q']:.3f}"
            ])
        
        models_to_show = [model] if model else MODELS
        for m in models_to_show:
            for corr, label in [('ratio', 'RM'), ('boosted_ratio', 'bRM')]:
                p = self.params.get(m, {}).get(corr)
                if p:
                    rows.append([
                        f"{m.upper()}+{label}",
                        f"{gv.mean(p['a0']):.4f}({gv.sdev(p['a0']):.4f})",
                        f"{gv.mean(p['E0']):.5f}({gv.sdev(p['E0']):.5f})",
                        f"{gv.mean(p['dE1']):.4f}({gv.sdev(p['dE1']):.4f})",
                        f"{p['chi2_dof']:.2f}",
                        f"{p['Q']:.3f}"
                    ])
        
        if rows:
            table = ax.table(cellText=rows, colLabels=headers,
                           loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
    
    def plot_benchmark_params(self, axes):
        """4 subplots showing a0, E0, a1, dE1 across Truth, ML+BC, ML+RM per model."""
        params = ['a0', 'E0', 'a1', 'dE1']
        colours = {'BC': 'steelblue', 'RM': 'green'}
        
        for ax, param in zip(axes.flat, params):
            truth_p = self.params.get('truth')
            if truth_p and param in truth_p:
                tv = gv.mean(truth_p[param])
                te = gv.sdev(truth_p[param])
                ax.axhline(tv, color='black', ls='--', lw=2, label='Truth')
                ax.axhspan(tv - te, tv + te, alpha=0.2, color='gray')
            
            x_pos = 0
            xticks, xlabels = [], []
            for model in MODELS:
                for corr, label in [('bias_corrected', 'BC'), ('ratio', 'RM')]:
                    p = self.params.get(model, {}).get(corr)
                    if p and param in p:
                        val = gv.mean(p[param])
                        err = gv.sdev(p[param])
                        ax.bar(x_pos, val, yerr=err, capsize=3, 
                               color=colours[label], alpha=0.7, width=0.8)
                        xticks.append(x_pos)
                        xlabels.append(f'{model[:3]}+{label}')
                        x_pos += 1
                x_pos += 0.5
            
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(param)
            ax.set_title(param)
            if param == 'a0':
                ax.legend(loc='upper right')
    
    def plot_parameter_comparison(self, ax, param: str):
        truth_params = self.params.get('truth')
        truth_val = truth_params.get(param) if truth_params else None
        
        vals, errs, labels = [], [], []
        for model in MODELS:
            model_params = self.params.get(model, {})
            p = model_params.get('bias_corrected') if model_params else None
            if p and param in p:
                labels.append(model.upper())
                vals.append(gv.mean(p[param]))
                errs.append(gv.sdev(p[param]))
        
        if not labels:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            return
        
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, capsize=5, color='steelblue', alpha=0.7)
        if truth_val is not None:
            ax.axhline(gv.mean(truth_val), color='black', ls='--', lw=2, label='Truth')
            ax.axhspan(gv.mean(truth_val) - gv.sdev(truth_val),
                      gv.mean(truth_val) + gv.sdev(truth_val),
                      alpha=0.2, color='gray')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(param)
        ax.legend()
    
    def plot_bias_comparison(self, ax, model: str, clean: bool = False, show_errors: bool = False):
        """Plot bias corrected vs uncorrected predictions for a model."""
        plot_data = [
            (self.correlators.get('truth'), 'Truth', 'black'),
            (self.correlators.get(model, {}).get('bias_corrected'), 'Bias Corrected', 'blue'),
            (self.correlators.get(model, {}).get('raw'), 'Raw (Uncorrected)', 'red'),
        ]
        
        for data, label, colour in plot_data:
            if data is None:
                continue
            
            mean = np.mean(data, axis=0)
            err = np.std(data, axis=0) / np.sqrt(len(data))
            t = np.arange(len(mean))
            
            if clean:
                mask = t % 2 == 1
                t, mean, err = t[mask], mean[mask], err[mask]
            
            if show_errors:
                ax.errorbar(t, np.log(np.abs(mean)), yerr=err/np.abs(mean),
                           fmt='o-', color=colour, label=label, 
                           ms=3, capsize=2, alpha=0.8)
            else:
                ax.plot(t, np.log(np.abs(mean)), 'o-', color=colour, label=label, 
                       ms=3, alpha=0.8)
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('ln|C(t)|', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_title(f'{model.upper()}: Bias Correction Effect', fontsize=11)
    
    def plot_ratio_comparison(self, ax, model: str, clean: bool = False, show_errors: bool = False):
        """Plot ratio method vs boosted ratio method for a model."""
        plot_data = [
            (self.correlators.get('truth'), 'Truth', 'black'),
            (self.correlators.get(model, {}).get('bias_corrected'), 'Bias Corrected', 'blue'),
            (self.correlators.get(model, {}).get('ratio'), 'Ratio Method', 'green'),
            (self.correlators.get(model, {}).get('boosted_ratio'), 'Boosted Ratio', 'orange'),
        ]
        
        for data, label, colour in plot_data:
            if data is None:
                continue
            
            mean = np.mean(data, axis=0)
            err = np.std(data, axis=0) / np.sqrt(len(data))
            t = np.arange(len(mean))
            
            if clean:
                mask = t % 2 == 1
                t, mean, err = t[mask], mean[mask], err[mask]
            
            if show_errors:
                ax.errorbar(t, np.log(np.abs(mean)), yerr=err/np.abs(mean),
                           fmt='o-', color=colour, label=label, 
                           ms=3, capsize=2, alpha=0.8)
            else:
                ax.plot(t, np.log(np.abs(mean)), 'o-', color=colour, label=label, 
                       ms=3, alpha=0.8)
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('ln|C(t)|', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_title(f'{model.upper()}: Ratio Methods', fontsize=11)
    
    def plot_noise_comparison(self, ax, sources: List[str]):
        """Plot noise-to-signal ratio comparison."""
        for src in sources:
            if src == 'truth':
                data = self.correlators.get('truth')
            else:
                data = self.correlators.get(src, {}).get('bias_corrected')
            
            if data is None:
                continue
            
            t, nsr = self.compute_noise_to_signal(data)
            ax.semilogy(t, nsr, 'o-', color=COLOURS.get(src, 'gray'), 
                       label=src.upper() if src != 'truth' else 'Truth', ms=3)
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('Noise/Signal', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, TIME_EXTENT)
        ax.set_title('Noise-to-Signal Ratio', fontsize=11)

    def plot_fit_parameters(self, fig) -> None:
        """Create 2x2 fit parameter comparison plot."""
        param_map = {
            r'$a_0$': 'a0',
            r'$E_0$': 'E0',
            r'$a_1$': 'a1', 
            r'$dE_1$': 'dE1',
        }
        
        axes = fig.subplots(2, 2)
        
        # Collect all entries
        plot_entries = []
        truth_p = self.params.get('truth')
        if truth_p:
            plot_entries.append(('Truth', COLOURS['truth'], 'o', truth_p))
        
        for model in MODELS:
            mp = self.params.get(model, {})
            bc_p = mp.get('bias_corrected')
            if bc_p:
                plot_entries.append((model.upper(), COLOURS.get(model, 'gray'), 'o', bc_p))
        
        # Plot parameters
        for ax, (display_name, param_key) in zip(axes.flat, param_map.items()):
            if truth_p and param_key in truth_p:
                tv = gv.mean(truth_p[param_key])
                te = gv.sdev(truth_p[param_key])
                ax.axhspan(tv - te, tv + te, alpha=0.3, color=COLOURS['truth'], zorder=0)
            
            for i, (label, color, marker, params) in enumerate(plot_entries):
                if param_key not in params:
                    continue
                val = gv.mean(params[param_key])
                err = gv.sdev(params[param_key])
                if err > 100 * abs(val) and abs(val) > 1e-10:
                    continue
                ax.errorbar(i, val, yerr=err, fmt=marker, color=color,
                           capsize=3, capthick=1.5, markersize=8, 
                           markeredgecolor='black', markeredgewidth=0.5)
            
            ax.set_ylabel(display_name, fontsize=12)
            ax.set_xticks([])
            ax.set_title(display_name, fontsize=12)
            
            # Wider y-range
            if truth_p and param_key in truth_p:
                tv = gv.mean(truth_p[param_key])
                te = gv.sdev(truth_p[param_key])
                margin = max(te * 10, abs(tv) * 0.3)
                ax.set_ylim(tv - margin, tv + margin)
        
        # Legend at bottom (only if we have entries)
        if plot_entries:
            legend_handles = []
            legend_labels = []
            for label, color, marker, _ in plot_entries:
                handle = plt.Line2D([0], [0], marker=marker, color=color, 
                                   linestyle='None', markersize=8,
                                   markeredgecolor='black', markeredgewidth=0.5)
                legend_handles.append(handle)
                legend_labels.append(label)
            
            fig.legend(legend_handles, legend_labels, loc='lower center',
                      ncol=len(plot_entries), fontsize=10, frameon=False)
    
    def plot_results_tables(self, fig) -> None:
        """Create page with two tables: BC models and Ratio Method models."""
        # Create two subplots for the two tables
        ax1, ax2 = fig.subplots(2, 1)
        ax1.axis('off')
        ax2.axis('off')
        
        truth_p = self.params.get('truth')
        
        # Table 1: Bias Corrected ML Models
        headers1 = ['Model', 'a₀', 'E₀', 'a₁', 'dE₁', 'χ²/dof', 'Q']
        rows1 = []
        
        if truth_p:
            rows1.append([
                'Truth',
                f"{gv.mean(truth_p['a0']):.4f}({gv.sdev(truth_p['a0']):.4f})",
                f"{gv.mean(truth_p['E0']):.5f}({gv.sdev(truth_p['E0']):.5f})",
                f"{gv.mean(truth_p['a1']):.4f}({gv.sdev(truth_p['a1']):.4f})",
                f"{gv.mean(truth_p['dE1']):.4f}({gv.sdev(truth_p['dE1']):.4f})",
                f"{truth_p['chi2_dof']:.2f}",
                f"{truth_p['Q']:.3f}"
            ])
        
        for model in MODELS:
            p = self.params.get(model, {}).get('bias_corrected')
            if p:
                rows1.append([
                    model.upper(),
                    f"{gv.mean(p['a0']):.4f}({gv.sdev(p['a0']):.4f})",
                    f"{gv.mean(p['E0']):.5f}({gv.sdev(p['E0']):.5f})",
                    f"{gv.mean(p['a1']):.4f}({gv.sdev(p['a1']):.4f})",
                    f"{gv.mean(p['dE1']):.4f}({gv.sdev(p['dE1']):.4f})",
                    f"{p['chi2_dof']:.2f}",
                    f"{p['Q']:.3f}"
                ])
        
        if rows1:
            ax1.set_title('Bias Corrected ML Models', fontsize=14, fontweight='bold', pad=20)
            table1 = ax1.table(cellText=rows1, colLabels=headers1,
                              loc='center', cellLoc='center')
            table1.auto_set_font_size(False)
            table1.set_fontsize(11)
            table1.scale(1.2, 2.0)
        
        # Table 2: Ratio Method + ML Models
        headers2 = ['Model', 'a₀', 'E₀', 'a₁', 'dE₁', 'χ²/dof', 'Q']
        rows2 = []
        
        if truth_p:
            rows2.append([
                'Truth',
                f"{gv.mean(truth_p['a0']):.4f}({gv.sdev(truth_p['a0']):.4f})",
                f"{gv.mean(truth_p['E0']):.5f}({gv.sdev(truth_p['E0']):.5f})",
                f"{gv.mean(truth_p['a1']):.4f}({gv.sdev(truth_p['a1']):.4f})",
                f"{gv.mean(truth_p['dE1']):.4f}({gv.sdev(truth_p['dE1']):.4f})",
                f"{truth_p['chi2_dof']:.2f}",
                f"{truth_p['Q']:.3f}"
            ])
        
        for model in MODELS:
            for corr, label in [('ratio', 'RM'), ('boosted_ratio', 'bRM')]:
                p = self.params.get(model, {}).get(corr)
                if p:
                    rows2.append([
                        f"{model.upper()}+{label}",
                        f"{gv.mean(p['a0']):.4f}({gv.sdev(p['a0']):.4f})",
                        f"{gv.mean(p['E0']):.5f}({gv.sdev(p['E0']):.5f})",
                        f"{gv.mean(p['a1']):.4f}({gv.sdev(p['a1']):.4f})",
                        f"{gv.mean(p['dE1']):.4f}({gv.sdev(p['dE1']):.4f})",
                        f"{p['chi2_dof']:.2f}",
                        f"{p['Q']:.3f}"
                    ])
        
        if rows2:
            ax2.set_title('Ratio Method + ML Models', fontsize=14, fontweight='bold', pad=20)
            table2 = ax2.table(cellText=rows2, colLabels=headers2,
                              loc='center', cellLoc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(10)
            table2.scale(1.2, 1.8)

    def generate_pdf(self, clean: bool = False, show_errors: bool = False) -> Path:
        # Ensure fits are run and data is loaded
        if not self.correlators:
            self.run_fits(verbose=True)
        
        output = self.results / f"{self.experiment}_physics_report.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output) as pdf:
            # Page 1: Log correlator comparison (all models vs truth)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nLog Correlator Comparison', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_log_correlator(ax, ['truth', model], clean, show_errors)
                ax.set_title(f'{model.upper()} vs Truth', fontsize=11)
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Noise-to-signal comparison
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle(f'{self.experiment}\nNoise-to-Signal Ratio', fontsize=14)
            self.plot_noise_comparison(ax, ['truth'] + MODELS)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Bias corrected vs uncorrected comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nBias Correction Effect', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_bias_comparison(ax, model, clean, show_errors)
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 4: Ratio method comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nRatio Methods Comparison', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_ratio_comparison(ax, model, clean, show_errors)
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 5: Fit parameter comparison
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nFit Parameters (Bias Corrected)', fontsize=14)
            self.plot_fit_parameters(fig)
            plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3, wspace=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 6: Results tables
            fig = plt.figure(figsize=(12, 14))
            fig.suptitle(f'{self.experiment}\nFit Results Summary', fontsize=16, fontweight='bold')
            self.plot_results_tables(fig)
            plt.subplots_adjust(top=0.92, hspace=0.4)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"PDF saved: {output}")
        return output
    
    def run(self, save_pdf: bool = True, clean: bool = False, verbose: bool = True):
        if not self.models_ready():
            print("Run missing models first.")
            return None
        
        self.run_fits(verbose)
        
        if save_pdf:
            return self.generate_pdf(clean)
        else:
            return self.params

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python physics_analysis.py <experiment> [--clean]")
        sys.exit(1)
    
    experiment = sys.argv[1]
    clean = '--clean' in sys.argv
    
    report = TwoPointReport(experiment)
    report.run(clean=clean)

if __name__ == "__main__":
    main()
