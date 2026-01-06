"""Bayesian fitting and PDF report for 2pt correlators."""

import json
import numpy as np
import gvar as gv
import corrfitter as cf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Dict, List, Optional
from .preliminary_analysis import find_plateau

TIME_EXTENT = 95
MODELS = ['cnn', 'gbr', 'mlp', 'transformer']
COLOURS = {'truth': '#1f77b4', 'cnn': '#ff7f0e', 'gbr': '#2ca02c', 'mlp': '#d62728', 'transformer': '#9467bd'}

# Fit configuration (found via systematic scanning)
FIT_CONFIG = {
    'bin_size': 4, 'tmin_truth': 14, 'tmin_ml': 20, 'tmax': 40,
    'svdcut': 0.01, 'n_normal': 2, 'n_oscillating': 0, 'prior_width': 0.4
}

# Model-specific optimal parameters
MODEL_FIT_CONFIG = {
    'cnn': {'bias_corrected': {'tmin': 22, 'width_scale': 0.3}, 'ratio': {'tmin': 24, 'width_scale': 0.6}},
    'gbr': {'bias_corrected': {'tmin': 20, 'width_scale': 0.4}, 'ratio': {'tmin': 20, 'width_scale': 0.4}},
    'mlp': {'bias_corrected': {'tmin': 24, 'width_scale': 1.0}, 'ratio': {'tmin': 30, 'width_scale': 0.6}},
    'transformer': {'bias_corrected': {'tmin': 24, 'width_scale': 0.5}, 'ratio': {'tmin': 24, 'width_scale': 0.5}},
}


def bin_samples(data: np.ndarray, bin_size: int) -> np.ndarray:
    """Bin jackknife samples to reduce correlations."""
    n = data.shape[0] // bin_size
    return np.array([np.mean(data[i*bin_size:(i+1)*bin_size], axis=0) for i in range(n)])


def build_prior(estimates: Dict, n_normal: int = 2, n_osc: int = 0, width_scale: float = 1.0) -> gv.BufferDict:
    """Build prior for corrfitter from plateau estimates."""
    prior = gv.BufferDict()
    lnE0 = estimates.get('lnE0', np.log(0.8))
    lna0 = estimates.get('lna0', -2.0)
    w = width_scale
    
    # Normal parity
    prior['log(a)'] = gv.gvar([lna0] + [lna0 - 2*i for i in range(1, n_normal)], [3*w]*n_normal)
    prior['log(dE)'] = gv.gvar([lnE0] + [np.log(0.2)]*(n_normal-1), [1*w] + [np.log(2)*w]*(n_normal-1))
    
    # Oscillating parity (if any)
    if n_osc > 0:
        lnao = estimates.get('lnao0', lna0 - 1)
        lnEo = estimates.get('lnEo0', lnE0)
        prior['log(ao)'] = gv.gvar([lnao] + [lnao - 2*i for i in range(1, n_osc)], [3*w]*n_osc)
        prior['log(dEo)'] = gv.gvar([lnEo] + [np.log(0.2)]*(n_osc-1), [1*w] + [np.log(2)*w]*(n_osc-1))
    
    return prior


def do_fit(correlator: np.ndarray, tmin: int, tmax: int, prior: gv.BufferDict,
           svdcut: float = 0.01, n_osc: int = 0) -> Optional[cf.CorrFitter]:
    """Fit correlator with corrfitter."""
    gv_data = gv.dataset.avg_data(correlator, bstrap=False)
    try:
        if n_osc > 0:
            model = cf.Corr2(datatag='2pt', tp=TIME_EXTENT, tmin=tmin, tmax=tmax,
                            a=('a', 'ao'), b=('a', 'ao'), dE=('dE', 'dEo'), s=(1.0, -1.0))
        else:
            model = cf.Corr2(datatag='2pt', tp=TIME_EXTENT, tmin=tmin, tmax=tmax, a='a', b='a', dE='dE')
        fitter = cf.CorrFitter(models=[model])
        return fitter.lsqfit(data={'2pt': gv_data}, prior=prior, svdcut=svdcut)
    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def extract_params(fit) -> Dict:
    """Extract E0, a0, dE1, a1, chi2/dof, Q from fit."""
    p = fit.p
    dE = p.get('log(dE)', gv.gvar([0], [1]))
    a = p.get('log(a)', gv.gvar([0], [1]))
    E0 = gv.exp(dE[0]) if hasattr(dE, '__len__') else gv.exp(dE)
    a0 = gv.exp(a[0]) if hasattr(a, '__len__') else gv.exp(a)
    dE1 = gv.exp(dE[1]) if hasattr(dE, '__len__') and len(dE) > 1 else gv.gvar(0, 1)
    a1 = gv.exp(a[1]) if hasattr(a, '__len__') and len(a) > 1 else gv.gvar(0, 1)
    return {'E0': E0, 'a0': a0, 'dE1': dE1, 'a1': a1, 'chi2_dof': fit.chi2/max(1, fit.dof), 'Q': fit.Q}


class TwoPointReport:
    """PDF report comparing truth vs ML model fits."""
    
    def __init__(self, experiment: str, project_root: Path = None):
        self.experiment = experiment
        self.root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.results = self.root / "results" / experiment
        self.data = self.root / "data"
        self.correlators, self.estimates, self.params, self.fits = {}, {}, {}, {}
    
    def models_ready(self) -> bool:
        missing = [m for m in MODELS if not (self.results / m / "bias_corrected" / "correlator_predictions.npy").exists()]
        if missing:
            print(f"Missing: {missing}")
        return not missing
    
    def load_truth(self) -> np.ndarray:
        exp_path = self.data / "experiments" / self.experiment
        meta = exp_path / "metadata.json"
        if meta.exists():
            with open(meta) as f:
                target = Path(json.load(f).get('target filename', '')).stem
            jk = self.data / "processed" / "jackknife_samples" / f"{target}_jackknife_samples.npy"
            if jk.exists():
                return np.load(jk)
        return np.load(exp_path / "test_data_y.npy")
    
    def load_ml(self, model: str, correction: str) -> Optional[np.ndarray]:
        paths = {
            'bias_corrected': self.results / model / "bias_corrected" / "correlator_predictions.npy",
            'ratio': self.results / model / "ratio_predictions" / "correlator_predictions.npy",
        }
        path = paths.get(correction)
        return np.load(path) if path and path.exists() else None
    
    def get_estimates(self, data: np.ndarray) -> Dict:
        """Get plateau estimates for priors."""
        mean = np.mean(data, axis=0)
        err = np.std(data, axis=0) / np.sqrt(len(data))
        return find_plateau(mean, err)
    
    def run_fits(self, verbose: bool = True):
        """Run Bayesian fits on truth and ML predictions."""
        cfg = FIT_CONFIG
        if verbose:
            print(f"Running fits with optimized configuration...")
            print(f"  bin_size={cfg['bin_size']}, tmin_truth={cfg['tmin_truth']}, tmin_ml={cfg['tmin_ml']}, tmax={cfg['tmax']}")
            print(f"  n_normal={cfg['n_normal']}, n_oscillating={cfg['n_oscillating']}, prior_width={cfg['prior_width']}")
        
        # Truth fit
        truth = self.load_truth()
        binned = bin_samples(truth, cfg['bin_size'])
        self.correlators['truth'] = truth
        self.estimates['truth'] = self.get_estimates(binned)
        
        if verbose:
            print(f"  Truth: {truth.shape[0]} samples -> {binned.shape[0]} bins")
            print("Fitting truth...")
        
        prior = build_prior(self.estimates['truth'], cfg['n_normal'], cfg['n_oscillating'], cfg['prior_width'])
        fit = do_fit(binned, cfg['tmin_truth'], cfg['tmax'], prior, cfg['svdcut'], cfg['n_oscillating'])
        if fit:
            self.fits['truth'], self.params['truth'] = fit, extract_params(fit)
            if verbose:
                E0 = self.params['truth']['E0']
                print(f"  Truth: E0={gv.mean(E0):.5f}({gv.sdev(E0):.5f}), chi2/dof={fit.chi2/fit.dof:.3f}, Q={fit.Q:.3f}")
        
        # ML fits
        for model in MODELS:
            if verbose:
                print(f"Fitting {model}...")
            self.correlators[model], self.params[model] = {}, {}
            
            for correction in ['bias_corrected', 'ratio']:
                data = self.load_ml(model, correction)
                if data is None:
                    continue
                
                binned = bin_samples(data, cfg['bin_size'])
                self.correlators[model][correction] = data
                
                # Model-specific config
                mcfg = MODEL_FIT_CONFIG.get(model, {}).get(correction, {})
                tmin = mcfg.get('tmin', cfg['tmin_ml'])
                width = mcfg.get('width_scale', cfg['prior_width'])
                
                prior = build_prior(self.get_estimates(binned), cfg['n_normal'], cfg['n_oscillating'], width)
                fit = do_fit(binned, tmin, cfg['tmax'], prior, cfg['svdcut'], cfg['n_oscillating'])
                
                if fit:
                    self.fits[f'{model}_{correction}'] = fit
                    self.params[model][correction] = extract_params(fit)
                    if verbose:
                        E0 = self.params[model][correction]['E0']
                        print(f"  {model} {correction}: E0={gv.mean(E0):.5f}({gv.sdev(E0):.5f}), chi2/dof={fit.chi2/fit.dof:.3f}, Q={fit.Q:.3f}")
    
    def _get_data(self, src: str, correction: str = 'bias_corrected') -> Optional[np.ndarray]:
        if src == 'truth':
            return self.correlators.get('truth')
        return self.correlators.get(src, {}).get(correction)
    
    def plot_log_correlator(self, ax, sources: List[str], clean: bool = False):
        for src in sources:
            data = self._get_data(src)
            if data is None:
                continue
            mean = np.mean(data, axis=0)
            t = np.arange(len(mean))
            if clean:
                mask = t % 2 == 1
                t, mean = t[mask], mean[mask]
            style = {'lw': 1.5, 'ms': 4 if src == 'truth' else 5, 'alpha': 0.6 if src == 'truth' else 0.9}
            ax.plot(t, np.log(np.abs(mean)), 'o-', color=COLOURS.get(src, 'gray'),
                   label=src.upper() if src != 'truth' else 'Truth', **style)
        ax.set_xlabel('t')
        ax.set_ylabel('ln|C(t)|')
        ax.legend(fontsize=9)
    
    def plot_noise_to_signal(self, ax, sources: List[str]):
        for src in sources:
            data = self._get_data(src)
            if data is None:
                continue
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            ax.semilogy(np.arange(len(mean)), np.abs(std/mean), 'o-', color=COLOURS.get(src, 'gray'),
                       label=src.upper() if src != 'truth' else 'Truth', ms=3)
        ax.set_xlabel('t')
        ax.set_ylabel('Noise/Signal')
        ax.legend()
        ax.set_xlim(0, TIME_EXTENT)
    
    def plot_comparison(self, ax, model: str, correction: str, clean: bool = False):
        """Plot truth vs model with given correction."""
        for src, label, color in [('truth', 'Truth', 'black'), (model, f'{model.upper()}', COLOURS.get(model, 'blue'))]:
            data = self._get_data(src, correction)
            if data is None:
                continue
            mean = np.mean(data, axis=0)
            t = np.arange(len(mean))
            if clean:
                mask = t % 2 == 1
                t, mean = t[mask], mean[mask]
            ax.plot(t, np.log(np.abs(mean)), 'o-', color=color, label=label, ms=3 if src != 'truth' else 5, alpha=0.8)
        ax.set_xlabel('t')
        ax.set_ylabel('ln|C(t)|')
        ax.legend()
    
    def plot_fit_parameters(self, fig):
        """2x2 parameter comparison with BC and RM for each model."""
        params = [('a0', r'$a_0$'), ('E0', r'$E_0$'), ('a1', r'$a_1$'), ('dE1', r'$dE_1$')]
        axes = fig.subplots(2, 2)
        
        entries = []
        truth_p = self.params.get('truth')
        if truth_p:
            entries.append(('Truth', COLOURS['truth'], 'o', truth_p))
        for model in MODELS:
            mp = self.params.get(model, {})
            if mp.get('bias_corrected'):
                entries.append((f'{model.upper()} BC', COLOURS.get(model), 'o', mp['bias_corrected']))
            if mp.get('ratio'):
                entries.append((f'{model.upper()} RM', COLOURS.get(model), 's', mp['ratio']))
        
        for ax, (key, label) in zip(axes.flat, params):
            if truth_p and key in truth_p:
                tv, te = gv.mean(truth_p[key]), gv.sdev(truth_p[key])
                ax.axhspan(tv - te, tv + te, alpha=0.3, color=COLOURS['truth'])
            
            for i, (name, color, marker, p) in enumerate(entries):
                if key not in p:
                    continue
                val, err = gv.mean(p[key]), gv.sdev(p[key])
                if err > 100 * abs(val) and abs(val) > 1e-10:
                    continue
                fc = color if 'BC' in name or name == 'Truth' else 'none'
                ax.errorbar(i, val, yerr=err, fmt=marker, color=color, capsize=3, ms=8,
                           markerfacecolor=fc, markeredgecolor=color, markeredgewidth=1.5)
            
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.set_xticks([])
        
        # Legend
        if entries:
            handles = [plt.Line2D([0], [0], marker=m, color=c, linestyle='None', ms=8,
                                 markerfacecolor=c if 'BC' in n or n == 'Truth' else 'none',
                                 markeredgecolor=c) for n, c, m, _ in entries]
            fig.legend(handles, [e[0] for e in entries], loc='lower center', ncol=min(5, len(entries)), fontsize=8)
    
    def plot_results_table(self, ax, correction: str = 'bias_corrected'):
        """Table of fit results."""
        ax.axis('off')
        headers = ['Model', 'a₀', 'E₀', 'χ²/dof', 'Q']
        rows = []
        
        def fmt(g):
            return f"{gv.mean(g):.5f}({gv.sdev(g):.5f})"
        
        if self.params.get('truth'):
            p = self.params['truth']
            rows.append(['Truth', fmt(p['a0']), fmt(p['E0']), f"{p['chi2_dof']:.2f}", f"{p['Q']:.3f}"])
        
        for model in MODELS:
            p = self.params.get(model, {}).get(correction)
            if p:
                label = model.upper() + ('+RM' if correction == 'ratio' else '')
                rows.append([label, fmt(p['a0']), fmt(p['E0']), f"{p['chi2_dof']:.2f}", f"{p['Q']:.3f}"])
        
        if rows:
            table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.0)
    
    def generate_pdf(self, clean: bool = False) -> Path:
        if not self.correlators:
            self.run_fits()
        
        output = self.results / f"{self.experiment}_physics_report.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output) as pdf:
            # Page 1: Log correlator comparison
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nLog Correlator Comparison', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_log_correlator(ax, ['truth', model], clean)
                ax.set_title(f'{model.upper()} vs Truth')
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Noise-to-signal
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(f'{self.experiment}\nNoise-to-Signal Ratio', fontsize=14)
            self.plot_noise_to_signal(ax, ['truth'] + MODELS)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Bias correction effect
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nBias Correction Effect', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_comparison(ax, model, 'bias_corrected', clean)
                ax.set_title(f'{model.upper()}: BC')
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 4: Ratio method
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nRatio Method', fontsize=14)
            for ax, model in zip(axes.flat, MODELS):
                self.plot_comparison(ax, model, 'ratio', clean)
                ax.set_title(f'{model.upper()}: RM')
                ax.set_xlim(0, TIME_EXTENT)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 5: Fit parameters
            fig = plt.figure(figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nFit Parameters', fontsize=14)
            self.plot_fit_parameters(fig)
            plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.3, wspace=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 6: Results tables
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(f'{self.experiment}\nFit Results', fontsize=14)
            ax1.set_title('Bias Corrected', fontsize=12)
            self.plot_results_table(ax1, 'bias_corrected')
            ax2.set_title('Ratio Method', fontsize=12)
            self.plot_results_table(ax2, 'ratio')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"PDF saved: {output}")
        return output
    
    def run(self, save_pdf: bool = True, clean: bool = False, verbose: bool = True):
        if not self.models_ready():
            print("Run missing models first.")
            return None
        self.run_fits(verbose)
        return self.generate_pdf(clean) if save_pdf else self.params


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python physics_analysis.py <experiment> [--clean]")
        sys.exit(1)
    TwoPointReport(sys.argv[1]).run(clean='--clean' in sys.argv)


