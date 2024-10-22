import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import retrieval_base.auxiliary_functions as af

import petitRADTRANS.nat_cst as nc

from tqdm import tqdm
import copy

def plot_envelopes(
        ax, y, x, x_indices=[(0,6),(1,5),(2,4)], 
        colors=['0.0','0.5','1.0'], median_kwargs=None
        ):
    
    patch = None
    for i, (idx_l, idx_u) in enumerate(x_indices):

        patch = ax.fill_betweenx(
            y=y, x1=x[idx_l], x2=x[idx_u], fc=colors[i], ec='none'
            )
    
    line = None
    if median_kwargs is not None:
        idx_m = median_kwargs.get('idx', 3)
        try:
            median_kwargs.pop('idx')
        except KeyError:
            pass

        line, = ax.plot(x[idx_m], y, **median_kwargs)

    return (line, patch, line)

def plot_condensation_curve(ax, pressure, species, FeH=0.0, ann_kwargs=None, **kwargs):

    coeffs = {
        'CaTiO3': [5.125, -0.277, -0.554], # Wakeford et al. (2017)
        'Fe': [5.44, -0.48, -0.48], # Visscher et al. (2010)
        'Mg2SiO4': [5.89, -0.37, -0.73], # Visscher et al. (2010)
        'MgSiO3': [6.26, -0.35, -0.70], # Visscher et al. (2010)
        'Cr': [6.576, -0.486, -0.486], # Morley et al. (2012)
        'KCl': [12.479, -0.879, -0.879], # Morley et al. (2012)
        'MnS': [7.45, -0.42, -0.84], # Visscher et al. (2006)
        'Na2S': [10.05, -0.72, -1.08], # Visscher et al. (2006)
        'ZnS': [12.52, -0.63, -1.26], # Visscher et al. (2006)
        'H2S': [86.49, -8.54, -8.54], # Visscher et al. (2006)
    }

    a, b, c = coeffs[species]
    y = a + b*np.log10(pressure) + c*FeH
    T = 1e4/y
    
    ax.plot(T, pressure, **kwargs, zorder=-2)

    if ann_kwargs is None:
        return
    
    y = ann_kwargs.get('y', 1e0)
    ann_kwargs['xy'] = (np.interp(np.log10(y),np.log10(pressure),T), y)
    print(ann_kwargs['xy'])
    ann_kwargs.pop('y')

    ax.annotate(**ann_kwargs, zorder=-1)

def indicate_ghost(
        ax, v_bary, show_text=True, 
        cmap=mpl.colors.LinearSegmentedColormap.from_list('',['0.9','1.0'])
        ):
    
    ghosts = np.array([
        [1119.44,1120.33], [1142.78,1143.76], [1167.12,1168.08], 
        [1192.52,1193.49], [1219.01,1220.04], [1246.71,1247.76], 
        [1275.70,1276.80], [1306.05,1307.15], [1337.98,1338.94], 
    ])
    ghosts += np.array([-0.1,+0.1])
    ghosts *= (1+v_bary/3e5)

    ylim = ax.get_ylim()
    height = np.abs(np.diff(ylim))
    for ghost_i in ghosts:
        # Plot the masked wavelengths due to the ghost signature        
        Z = np.abs(np.linspace(*ghost_i, 20) - ghost_i.mean())
        ax.imshow(
            Z.reshape(-1,1).T, cmap=cmap, vmin=0, vmax=Z.max(), 
            extent=[*ghost_i, *ylim], interpolation='bilinear', 
            aspect='auto', zorder=-1
            )
        if show_text:
            ax.annotate(
                'ghost', xy=(ghost_i.mean(),ylim[0]+0.07*height), rotation=90, 
                ha='center', va='bottom', fontsize=9, color='0.4'
            )

def indicate_lines(ax, x, y, label, label_y=None):

    X = np.array([x[0],x[0],x[1],x[1]])
    Y = np.array([y[0],y[1],y[1],y[0]])
    ax.plot(X, Y, c='k', lw=1, transform=ax.get_xaxis_transform())

    text_kwargs = dict(ha='center', va='center', fontsize=12)
    if label_y is None:
        label_y = y[1]
        text_kwargs['bbox'] = {'boxstyle':'square', 'ec':'none', 'fc':'w'}

    ax.text(x=x.mean(), y=label_y, s=label, transform=ax.get_xaxis_transform(), **text_kwargs)

def get_cmap(colors=['#4A0C0C','#fff0e6']):
    cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', colors)
    cmap.set_bad('w'); cmap.set_over('w'); cmap.set_under('k')
    return cmap

def get_color(
        idx, colors=['#580062','#006909','#c12d2d','#5588D0','#e1a722']
        ):
    return colors[idx]

class high_pass_filter:

    def __init__(self, window_length=301, polyorder=2, mode='nearest', axis=-1):

        self.kwargs = dict(
            window_length=window_length, 
            polyorder=polyorder, 
            mode=mode, axis=axis
        )
    
    def __call__(self, flux):

        mask_isfinite = np.isfinite(flux)
        lp_flux = np.nan * flux.copy()
        
        # Apply Savitzky-Golay filter to remove broad structure
        from scipy.signal import savgol_filter
        lp_flux[mask_isfinite] = savgol_filter(
            flux[mask_isfinite], **self.kwargs
        )
        
        return flux - lp_flux

def convert_CCF_to_SNR(rv, CCF, rv_sep=100):
    # Convert the cross-correlation function to a S/N function
    rv_mask = np.abs(rv) > rv_sep

    mean_CCF = np.mean(CCF[rv_mask])
    std_CCF  = np.std(CCF[rv_mask])

    return (CCF - mean_CCF) / std_CCF

def convert_mf_dict_to_VMR_dict(Chem, relative_to_key=None):

    VMR = {}
    for pRT_name_i, mass_fractions_i in Chem.mass_fractions_posterior.items():
        #print(pRT_name_i)
        if pRT_name_i == 'MMW':
            continue
        
        # Read mass of current species
        mask = (Chem.species_info.pRT_name == pRT_name_i)
        species_i = Chem.species_info.index[mask].tolist()[0]

        mass_i = Chem.read_species_info(species=species_i, info_key='mass')
        
        # Convert from mass fraction to VMR
        VMR[species_i] = Chem.mass_fractions_posterior['MMW'] / mass_i * mass_fractions_i
    
    if relative_to_key is None:
        return VMR

    VMR_rel = VMR[relative_to_key]
    VMR = {species_i: VMR_i/VMR_rel for species_i, VMR_i in VMR.items()}
    return VMR

class RetrievalResults:
    
    def __init__(
            self, prefix, m_set='K2166_cloudy', w_set='K2166', low_memory=True, load_posterior=False
            ):

        self.prefix = prefix
        self.m_set  = m_set
        self.w_set  = w_set

        # Remove attributes after running functions
        self.low_memory = low_memory

        try:
            self.n_params = self._load_object('LogLike').N_params
                
            import pymultinest
            # Set-up analyzer object
            analyzer = pymultinest.Analyzer(
                n_params=self.n_params, outputfiles_basename=prefix
                )
            stats = analyzer.get_stats()
            self.ln_Z = stats['nested importance sampling global log-evidence']
            #self.ln_Z = stats['nested sampling global log-evidence']
            print(list(stats.keys()))
            print(stats['nested sampling global log-evidence'])
            print(stats['nested importance sampling global log-evidence'])
            print(stats['global evidence'])

            if load_posterior:
                self.posterior = analyzer.get_equal_weighted_posterior()
                self.posterior = self.posterior[:,:-1]

            # Read the parameters of the best-fitting model
            self.bestfit_values = np.array(stats['modes'][0]['maximum a posterior'])

            import json
            # Read the best-fitting parameters
            with open(self.prefix+'data/bestfit.json') as f:
                self.bestfit_params = json.load(f)['params']
                
        except:
            return

    def _load_object(self, name, bestfit_prefix=True, m_set=None):

        # Model setting
        setting = m_set
        if m_set is None:
            setting = self.m_set
        if name == 'd_spec':
            setting = self.w_set # Use wavelength-setting instead

        file_name = f'{self.prefix}data/bestfit_{name}_{setting}.pkl'
        if not bestfit_prefix:
            file_name = file_name.replace('bestfit_', '')

        try:
            return af.pickle_load(file_name)
        except FileNotFoundError:
            pass

        file_name = f'{self.prefix}data/bestfit_{name}.pkl'
        if not bestfit_prefix:
            file_name = file_name.replace('bestfit_', '')

        if name in ['LogLike', 'Cov']:
            return af.pickle_load(self.prefix+f'data/bestfit_{name}.pkl')
        
    def _load_objects_as_attr(self, names):

        if isinstance(names, str):
            names = [names]

        for name in names:
            if hasattr(self, name):
                continue
            setattr(self, name, self._load_object(name))

    def compare_evidence(self, ln_Z_other):
        '''Convert log-evidences of two models to a sigma confidence level'''

        from scipy.special import lambertw as W
        from scipy.special import erfcinv

        ln_B = self.ln_Z - ln_Z_other
        B = np.exp(ln_B)
        p = np.real(np.exp(W((-1.0/(B*np.exp(1))),-1)))
        sigma = np.sqrt(2)*erfcinv(p)

        _ln_B = ln_Z_other - self.ln_Z
        _B = np.exp(_ln_B)
        _p = np.real(np.exp(W((-1.0/(_B*np.exp(1))),-1)))
        _sigma = np.sqrt(2)*erfcinv(_p)

        print('Current vs. given: ln(B)={:.2f} | sigma={:.2f}'.format(ln_B, sigma))
        print('Given vs. current: ln(B)={:.2f} | sigma={:.2f}'.format(_ln_B, _sigma))
        return B, sigma
    
    def print_retrieved_values(self, indices, print_all=False):

        q = np.array([
            0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
            0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
            ])

        for i in indices:
            
            median = np.median(self.posterior[:,i])
            
            up  = np.quantile(self.posterior[:,i], q=q[4])
            low = np.quantile(self.posterior[:,i], q=q[2])

            up  -= median
            low -= median

            if print_all:
                print(
                    '{:.1f}'.format(median)+'^{+'+'{:.1f}'.format(up)+'}_{'+'{:.1f}'.format(low)+'}', 
                    '{:.2f}'.format(median)+'^{+'+'{:.2f}'.format(up)+'}_{'+'{:.2f}'.format(low)+'}', 
                    '{:.3f}'.format(median)+'^{+'+'{:.3f}'.format(up)+'}_{'+'{:.3f}'.format(low)+'}'
                    )
                continue

            if (np.round(up, 1) != 0.) or (np.round(low, 1) != 0.):
                print('{:.1f}'.format(median)+'^{+'+'{:.1f}'.format(up)+'}_{'+'{:.1f}'.format(low)+'}')
            elif (np.round(up, 2) != 0.) or (np.round(low, 2) != 0.):
                print('{:.2f}'.format(median)+'^{+'+'{:.2f}'.format(up)+'}_{'+'{:.2f}'.format(low)+'}')
            else:
                print('{:.3f}'.format(median)+'^{+'+'{:.3f}'.format(up)+'}_{'+'{:.3f}'.format(low)+'}')
                
    def get_mean_scaled_uncertainty(self, order_or_det='order'):

        LogLike = self._load_object('LogLike', bestfit_prefix=True)
        Cov = self._load_object('Cov', bestfit_prefix=True)

        sigma = np.zeros((LogLike.n_orders,LogLike.n_dets), dtype=np.float64)
        
        for i in range(LogLike.n_orders):
            diag_i = []
            for j in range(LogLike.n_dets):
                
                if Cov[i,j] is None:
                    diag_ij = [np.nan]*2048
                else:
                    # First column of banded matrix is diagonal
                    diag_ij = Cov[i,j].cov[0] * LogLike.s2[i,j]
                diag_i.append(diag_ij)
                
                if order_or_det == 'det':
                    # Get the mean uncertainty per detector
                    sigma[i,j] = np.nanmean(np.sqrt(diag_ij))

            if order_or_det == 'order':
                diag_i = np.concatenate(diag_i)
                
                # Get the mean uncertainty per order
                sigma[i,:] = np.nanmean(np.sqrt(diag_i))

        return sigma
    
    def get_Rot(self, pRT_atm=None):

        if pRT_atm is None:
            pRT_atm = self._load_object('pRT_atm_broad', bestfit_prefix=False)
        
        return copy.copy(pRT_atm.Rot)
    
    def get_model_spec(
            self, is_local=False, m_set=None, 
            line_species_to_exclude=None, 
            line_species_to_include=None, 
            return_pRT_atm=False, 
            return_wave_flux=False
            ):
        
        # Load the necessary objects
        self._load_objects_as_attr(['Chem', 'PT'])
        pRT_atm = self._load_object(
            'pRT_atm_broad', bestfit_prefix=False, m_set=m_set
            )

        # Change the parameters for a local, un-broadened model
        if m_set is None:
            m_set = self.m_set
        params = copy.copy(self.bestfit_params[m_set])

        mf = copy.deepcopy(self.Chem.mass_fractions)

        if is_local:
            params['vsini'] = 0
            #params['rv'] = 0

            # Remove any spots/bands
            keys_to_delete = [
                'epsilon_lat', 'lat_band', 'lon_band', 'r_spot', 'lat_spot', 
                'is_within_patch', 'is_outside_patch', 
                'is_in_band', 'is_not_in_band', 
                'is_in_spot', 'is_not_in_spot', 
                ]
            for key in keys_to_delete:
                params.pop(key, None)

        # Change the local abundances
        if line_species_to_include is not None:
            for key_i in mf.keys():
                if key_i in ['MMW', 'H2', 'He']:
                    continue
                if key_i == line_species_to_include:
                    print(f'Generating a model with only {line_species_to_include}')
                    continue
                # Set abundances of other species to 0
                mf[key_i] *= 0
        
        if line_species_to_exclude is not None:
            for key_i in mf.keys():
                if key_i in ['MMW', 'H2', 'He']:
                    continue
                if key_i == line_species_to_exclude:
                    print(f'Generating a model w/o {line_species_to_exclude}')
                    # Set abundance of this species to 0
                    mf[key_i] *= 0

        # Only consider a single model
        pRT_atm.sum_m_spec = False

        # Update the pRT model
        m_spec = pRT_atm(
            mf, self.PT.temperature, params, 
            get_contr=False, get_full_spectrum=True
            )
        
        wave_pRT_grid = copy.copy(pRT_atm.wave_pRT_grid)
        flux_pRT_grid = copy.copy(pRT_atm.flux_pRT_grid)

        if self.low_memory:
            del self.Chem, self.PT, m_spec

        if return_wave_flux:
            del pRT_atm
            return wave_pRT_grid, flux_pRT_grid

        if return_pRT_atm:
            return wave_pRT_grid, flux_pRT_grid, self.get_Rot(pRT_atm=pRT_atm), pRT_atm
        
        return wave_pRT_grid, flux_pRT_grid, self.get_Rot(pRT_atm=pRT_atm) # Rotation model
    
    def get_int_contr_em(self, m_set=None):

        pRT_atm_broad = self._load_object(
            'pRT_atm_broad', bestfit_prefix=False, m_set=m_set
            )
        n_atm_layers = len(pRT_atm_broad.pressure)
        pRT_wave = [nc.c*1e7/atm_i.freq for atm_i in pRT_atm_broad.atm]
        del pRT_atm_broad

        d_spec  = self._load_object('d_spec', bestfit_prefix=False)
        LogLike = self._load_object('LogLike', bestfit_prefix=True)

        int_contr_em_per_order = np.zeros((d_spec.n_orders, n_atm_layers))
        # Loop over orders
        for i in range(d_spec.n_orders):
            
            wave_i = d_spec.wave[i].flatten()
            flux_i = LogLike.m_flux_phi[i].flatten()
            
            wave_i = wave_i[np.isfinite(flux_i)]
            flux_i = flux_i[np.isfinite(flux_i)]

            pRT_wave_i = pRT_wave[i]

            contr_em_i = np.load(self.prefix+f'data/contr_em_order{i}_{m_set}.npy')

            # Loop over atmospheric layers
            for j in range(n_atm_layers):
                # Rebin to same wavelength grid
                contr_em_ij = np.interp(wave_i, xp=pRT_wave_i, fp=contr_em_i[j])

                # Spectrally weighted integration
                integral1 = np.trapz(wave_i*flux_i*contr_em_ij, wave_i)
                integral2 = np.trapz(wave_i*flux_i, wave_i)

                int_contr_em_per_order[i,j] = integral1/integral2

        del d_spec, LogLike, pRT_wave

        int_contr_em = np.sum(int_contr_em_per_order, axis=0)
        return int_contr_em, int_contr_em_per_order
    
    def get_grey_cloud_opacity(self, keys_indices, m_set=None, N_fine_pressure=200):

        # Load the necessary objects
        Cloud = self._load_object(
            'pRT_atm', bestfit_prefix=False, m_set=m_set
            ).Cloud

        if Cloud.get_opacity is None:
            # Not a parameterised cloud
            return

        fine_pressure = np.logspace(
            np.log10(Cloud.pressure.min()), np.log10(Cloud.pressure.max()), 
            N_fine_pressure
            )
    
        opa_posterior, opa_posterior_fine = [], []
        # Loop over samples in posterior
        for posterior_i in self.posterior:
            
            # Add parameters to a dictionary
            params = {}
            for key_i, idx_i in keys_indices.items():

                if key_i.startswith('log_'):
                    params[key_i.replace('log_','')] = 10**posterior_i[idx_i]
                else:
                    params[key_i] = posterior_i[idx_i]

            # Update Cloud instance with sample
            Cloud(params=params, mass_fractions=None)

            # Get opacity at 1 micron
            opa_i = Cloud.cloud_opacity(
                wave_micron=np.array([1.0]), pressure=Cloud.pressure
                )
            
            if len(keys_indices) == 3:
                opa_fine_i = np.zeros_like(fine_pressure)
                
                mask_cl = (fine_pressure < params['P_base_gray'])
                opa_fine_i[mask_cl] = params['opa_base_gray'] * (
                    fine_pressure[mask_cl]/params['P_base_gray']
                    )**params['f_sed_gray']
                opa_fine_i = opa_fine_i[None,:]
                
            opa_posterior.append(opa_i[0])
            opa_posterior_fine.append(opa_fine_i[0])

        q = np.array([
            0.5-0.997/2, 0.5-0.95/2, 0.5-0.68/2, 0.5, 
            0.5+0.68/2, 0.5+0.95/2, 0.5+0.997/2
            ])
        opa_envelope      = np.quantile(opa_posterior, q=q, axis=0)
        opa_envelope_fine = np.quantile(opa_posterior_fine, q=q, axis=0)

        # Opacity with best-fitting parameters
        Cloud(params=self.bestfit_params[self.m_set], mass_fractions=None)
        opa_bestfit = Cloud.cloud_opacity(
            wave_micron=np.array([1.0]), pressure=Cloud.pressure
            )[0]

        return opa_posterior, opa_envelope, opa_posterior_fine, opa_envelope_fine, fine_pressure, opa_bestfit

    def get_example_line_profile(self, m_res=1e6, T=1200, P=1, mass=(2*1+16), return_Rot=False):

        k_B = 1.3807e-16    # cm^2 g s^-2 K^-1
        c   = 2.99792458e10 # cm s^-1
        
        amu = 1.66054e-24   # g
        mass *= amu
        
        # Load the necessary objects
        if not hasattr(self, 'd_spec'):
            self.d_spec = self._load_object('d_spec', bestfit_prefix=False)

        profile_wave_cen = np.nanmean(self.d_spec.wave)
        delta_wave       = profile_wave_cen / m_res # Use model resolution get wlen-spacing
        profile_wave     = profile_wave_cen + np.arange(-5, 5+1e-6, delta_wave)

        # In wavenumber (cm^-1)
        profile_wn_cen = 1e7 / profile_wave_cen
        profile_wn     = 1e7 / profile_wave

        # Broadening parameters
        gamma_G = np.sqrt((2*k_B*T)/mass) * profile_wn_cen/c
        gamma_L = 0.06 # Made up

        # Use a Voigt profile
        from scipy.special import wofz as Faddeeva

        u = (profile_wn - profile_wn_cen) / gamma_G
        a = gamma_L / gamma_G

        profile_flux = Faddeeva(u + 1j*a).real
        profile_flux = 1 - profile_flux/np.max(profile_flux) # Normalise

        _, _, Rot, pRT_atm = self.get_model_spec(is_local=False, return_pRT_atm=True)
        int_intensities  = np.sum(pRT_atm.atm[0].flux_mu, axis=-1)
        #int_intensities  = pRT_atm.atm[0].flux_mu[:,100]
        int_intensities /= np.sum(int_intensities) # Normalize
        #int_intensities /= np.max(int_intensities) # Normalize
        del pRT_atm

        # Scale the profile with the integrated mu-dependent intensities
        flux_mu = profile_flux[None,:]*int_intensities[:,None]

        params_vsini_0 = {'vsini':0}
        Rot.get_brightness(params=params_vsini_0)
        _, profile_flux_vsini_0 = Rot(
            wave=profile_wave, flux=flux_mu, 
            params=params_vsini_0, get_scaling=True, 
            )

        # Rotational broadening
        Rot.get_brightness(params=self.bestfit_params[self.m_set])
        _, profile_flux_broad = Rot(
            wave=profile_wave, flux=flux_mu, 
            params=self.bestfit_params[self.m_set], 
            get_scaling=True, 
            )
        #profile_flux_broad /= np.pi # Normalise

        # Normalise
        profile_flux_broad   /= np.nanmax(profile_flux_broad)
        #profile_flux_broad   /= profile_flux_vsini_0.max()
        profile_flux_vsini_0 /= profile_flux_vsini_0.max()

        # Instrumental broadening
        profile_flux_vsini_0 = self.d_spec.instr_broadening(
            wave=profile_wave, flux=profile_flux_vsini_0, 
            out_res=self.d_spec.resolution, in_res=m_res
            )
        profile_flux_broad = self.d_spec.instr_broadening(
            wave=profile_wave, flux=profile_flux_broad, 
            out_res=self.d_spec.resolution, in_res=m_res
            )
        
        if self.low_memory:
            del self.d_spec

        if return_Rot:
            return profile_wave, profile_flux_vsini_0, profile_flux_broad, Rot
        
        return profile_wave, profile_flux_vsini_0, profile_flux_broad
    
    def get_telluric_CCF(
            self, wave_local, flux_local, 
            rv=np.arange(-300,300+1e-6,0.5), 
            high_pass={'tell_res': high_pass_filter(), 'm_res': high_pass_filter()}, 
            orders_to_include=None, 
            path_data='../data/', 
            tell_threshold=0.7, 
            **kwargs
            ):

        # Load the necessary objects
        self._load_objects_as_attr(['Cov', 'LogLike'])
        self.d_spec = self._load_object('d_spec', bestfit_prefix=False)

        # Load the telluric standard observation
        _, flux_std, err_std = np.loadtxt(f'{path_data}Luhman_16_std_K.dat').T
        #self.tell_wave, transm_mf = np.loadtxt(f'{path_data}Luhman_16_std_K_molecfit_transm.dat').T
        #_, continuum = np.loadtxt(f'{path_data}Luhman_16_std_K_molecfit_continuum.dat').T
        self.tell_wave, transm_mf = np.loadtxt(f'{path_data}Luhman_16_std_K_molecfit_transm_new.dat').T
        _, continuum = np.loadtxt(f'{path_data}Luhman_16_std_K_molecfit_continuum_new.dat').T

        # Remove the continuum and tellurics
        flux_std /= continuum
        err_std  /= continuum

        self.tell_res = np.ones_like(flux_std) * np.nan
        self.tell_err = np.ones_like(err_std) * np.nan
        
        from scipy.ndimage import minimum_filter
        mask_tell = (transm_mf > tell_threshold)
        #mask_tell = minimum_filter(mask_tell, size=30)

        self.tell_res[mask_tell] = (flux_std/transm_mf)[mask_tell]
        #self.tell_res[mask_tell] = (flux_std)[mask_tell]
        self.tell_err[mask_tell] = (err_std/transm_mf)[mask_tell]

        #self.tell_res /= self.tell_res
        #self.tell_err /= self.tell_res

        # Reshape the arrays
        mask_wave = (self.tell_wave > self.d_spec.wave.min()-10) & \
            (self.tell_wave < self.d_spec.wave.max()+10)
        self.tell_wave = self.tell_wave[mask_wave].reshape(self.d_spec.wave.shape)
        self.tell_res  = self.tell_res[mask_wave].reshape(self.d_spec.wave.shape)
        self.tell_err  = self.tell_err[mask_wave].reshape(self.d_spec.wave.shape)

        #self.tell_res -= 1

        transm_mf = transm_mf[mask_wave].reshape(self.d_spec.wave.shape)

        # Mask the intrinsic hydrogen (H I) lines
        mask_lines = np.zeros_like(self.tell_res, dtype=bool)

        mask_lines = mask_lines | ((self.tell_wave > 2157.6) & (self.tell_wave < 2174.3))
        mask_lines = mask_lines | ((self.tell_wave > 1944-5) & (self.tell_wave < 1944+5))
        mask_lines = mask_lines | ((self.tell_wave > 2166-5) & (self.tell_wave < 2166+5))
        mask_lines = mask_lines | ((self.tell_wave > 2470.0-3) & (self.tell_wave < 2470.0+3))
        mask_lines = mask_lines | ((self.tell_wave > 2449.0-3) & (self.tell_wave < 2449.0+3))
        mask_lines = mask_lines | ((self.tell_wave > 2431.4-3) & (self.tell_wave < 2431.4+3))

        self.tell_res[mask_lines] = np.nan

        self.tell_res[~self.d_spec.mask_isfinite] = np.nan

        if high_pass.get('tell_res') is not None:
            # Apply a high-pass filter
            self.tell_res = high_pass.get('tell_res')(self.tell_res)

        # Shift the telluric residuals to the planet's restframe

        # Correct for the barycentric velocity
        #self.tell_wave = self.tell_wave * (1+(self.d_spec.v_bary)/(nc.c*1e-5))
            
        fig, ax = plt.subplots(figsize=(12,2.5))
        #ax.plot(self.tell_wave, flux_std, c='k', lw=1, alpha=0.5)
        ax.plot(self.tell_wave.flatten(), self.tell_res.flatten(), c='k', lw=1)
        #ax.set(xlim=(2320,2370), ylim=(0.95,1.05))
        #ax.set(ylim=(-0.05,0.05))
        plt.show()

        # Perform the cross-correlation
        CCF = np.nan * np.ones((len(rv), self.d_spec.n_orders, self.d_spec.n_dets))
        CCF_mf = np.nan * np.ones((len(rv), self.d_spec.n_orders, self.d_spec.n_dets))

        for i, rv_i in enumerate(tqdm(rv)):

            # Center the model in the telluric rest-frame
            rv_i -= self.bestfit_params[self.m_set]['rv']
            #rv_i += self.vtell
        
            for j in range(self.d_spec.n_orders):

                # Shift the model spectrum
                m_wave_local = np.copy(wave_local[j]) * (1+rv_i/(nc.c*1e-5))
                m_flux_local = np.copy(flux_local[j])
                
                for k in range(self.d_spec.n_dets):

                    # Ignore the nans
                    mask_jk = self.d_spec.mask_isfinite[j,k] & \
                        np.isfinite(self.tell_res[j,k])
                    if not mask_jk.any():
                        continue

                    m_res_local_jk = m_flux_local.copy()

                    m_wave_mf = np.copy(self.tell_wave[j,k]) * (1+rv[i]/(nc.c*1e-5))
                    m_res_mf_jk = np.copy(transm_mf[j,k])

                    # Interpolate onto the data wavelength-grid
                    m_res_local_jk = np.interp(
                        self.tell_wave[j,k], xp=m_wave_local, fp=m_res_local_jk
                        )
                    m_res_local_jk *= self.LogLike.phi[j,k,0]

                    m_res_mf_jk = np.interp(
                        self.tell_wave[j,k], xp=m_wave_mf, fp=m_res_mf_jk, 
                        left=np.nan, right=np.nan
                        )
                    m_res_mf_jk *= self.LogLike.phi[j,k,0]

                    if high_pass.get('m_res') is not None:
                        # Apply a high-pass filter
                        m_res_local_jk = high_pass.get('m_res')(m_res_local_jk)
                        m_res_mf_jk = high_pass.get('m_res')(m_res_mf_jk)

                    # Compute the cross-correlation coefficient
                    CCF[i,j,k] = np.dot(
                        m_res_local_jk[mask_jk], 
                        self.tell_res[j,k,mask_jk] / self.tell_err[j,k,mask_jk]**2
                        #1/self.LogLike.s2[j,k] * \
                        #    self.Cov[j,k].solve(self.tell_res[j,k,mask_jk])
                        )
                    
                    mask_jk = mask_jk & np.isfinite(m_res_mf_jk)
                    CCF_mf[i,j,k] = np.dot(
                        m_res_mf_jk[mask_jk], 
                        self.tell_res[j,k,mask_jk] / self.tell_err[j,k,mask_jk]**2
                        #1/self.LogLike.s2[j,k] * \
                        #    self.Cov[j,k].solve(self.tell_res[j,k,mask_jk])
                        )

                    '''
                    if rv[i] != 0.:
                        continue

                    fig, ax = plt.subplots(figsize=(12,5))
                    #ax.plot(self.tell_wave, flux_std, c='k', lw=1, alpha=0.5)
                    ax.plot(self.tell_wave[j,k,mask_jk], self.tell_res[j,k,mask_jk], c='k', lw=1)
                    #ax.set(xlim=(2320,2370), ylim=(0.95,1.05))
                    ax.set(ylim=(0.95,1.05))
                    plt.show()
                    '''
                    
        if self.low_memory:
            del self.d_spec, self.Cov, self.LogLike, self.tell_wave, self.tell_res

        if orders_to_include is None:
            CCF_sum = np.nansum(CCF, axis=(1,2))
        else:
            CCF_sum = CCF[:,orders_to_include,:].sum(axis=(1,2))
        CCF_SNR = convert_CCF_to_SNR(rv, CCF_sum, **kwargs)

        if orders_to_include is None:
            CCF_sum = np.nansum(CCF_mf, axis=(1,2))
        else:
            CCF_sum = CCF_mf[:,orders_to_include,:].sum(axis=(1,2))
        CCF_SNR_mf = convert_CCF_to_SNR(rv, CCF_sum, **kwargs)

        return rv, CCF, CCF_SNR, CCF_mf, CCF_SNR_mf
    
    def add_patch_to_Rot(self, Rot, Rot_spot):

        # Fill in the patch with another Rot-instance
        mask_patch = Rot_spot.included_segments
        Rot.int_flux[mask_patch] = Rot_spot.int_flux[mask_patch]
    
    def get_CCF(
            self, wave_local, flux_local, 
            rv=np.arange(-300,300+1e-6,0.5), 
            high_pass={'m_res': high_pass_filter()}, 
            orders_to_include=None, 
            model_to_subtract_from_d_res=None, 
            model_to_subtract_from_m_res=None, 
            plot=False, 
            **kwargs
            ):

        # Load the necessary objects
        self._load_objects_as_attr(['Cov', 'LogLike'])
        self.d_spec = self._load_object('d_spec', bestfit_prefix=False)

        # Barycentric velocity (already corrected for)
        self.vtell = self.d_spec.v_bary - self.bestfit_params[self.m_set]['rv']

        # Cross-correlation
        CCF = np.nan * np.ones(
            (len(rv), self.d_spec.n_orders, self.d_spec.n_dets)
            )
    
        d_wave = np.copy(self.d_spec.wave)
        d_res  = np.copy(self.d_spec.flux)

        #fig, ax = plt.subplots(
        #    figsize=(12,3*self.d_spec.n_orders), 
        #    nrows=self.d_spec.n_orders
        #    )

        if model_to_subtract_from_d_res in ['m_flux_phi', 'complete']:
            # Residual wrt the complete model 
            d_res -= np.copy(self.LogLike.m_flux_phi)
        elif model_to_subtract_from_d_res is not None:
            # Subtract a model from the data
            for j in range(self.d_spec.n_orders):
                for k in range(self.d_spec.n_dets):

                    model_to_subtract_from_d_res_jk = np.interp(
                        d_wave[j,k], xp=wave_local[j], 
                        fp=model_to_subtract_from_d_res[j]*self.LogLike.phi[j,k,0]
                    )

                    #ax[j].plot(d_wave[j,k], d_res[j,k], c='k', lw=0.8)
                    #ax[j].plot(
                    #    d_wave[j,k], model_to_subtract_from_d_res_jk, c='C1', lw=1.2
                    #    )

                    d_res[j,k] -= model_to_subtract_from_d_res_jk

                #ax[j].set(xlim=(d_wave[j,:].min()-0.3,d_wave[j,:].max()+0.3))
        
        #plt.tight_layout()
        #plt.show()

        if high_pass.get('d_res') is not None:
            # Apply a high-pass filter
            d_res = high_pass.get('d_res')(d_res)

        if plot:
            fig, ax = plt.subplots(
                figsize=(12,3*self.d_spec.n_orders), 
                nrows=self.d_spec.n_orders, sharey=True
                )

        for i, rv_i in enumerate(tqdm(rv)):

            for j in range(self.d_spec.n_orders):

                # Shift the model spectrum
                m_wave_local = np.copy(wave_local[j]) * (1+rv_i/(nc.c*1e-5))
                m_flux_local = np.copy(flux_local[j])

                for k in range(self.d_spec.n_dets):

                    # Ignore the nans
                    mask_jk = self.d_spec.mask_isfinite[j,k]
                    if not mask_jk.any():
                        continue

                    m_res_local_jk = m_flux_local.copy()

                    # Subtract the global model from the template
                    if model_to_subtract_from_m_res is not None:
                        m_res_local_jk -= model_to_subtract_from_m_res[j]

                    # Interpolate onto the data wavelength-grid
                    m_res_local_jk = np.interp(
                        d_wave[j,k], xp=m_wave_local, fp=m_res_local_jk
                        )
                    m_res_local_jk *= self.LogLike.phi[j,k,0]

                    if high_pass.get('m_res') is not None:
                        # Apply a high-pass filter
                        m_res_local_jk = high_pass.get('m_res')(m_res_local_jk)
                    
                    # Compute the cross-correlation coefficient
                    CCF[i,j,k] = np.dot(
                        m_res_local_jk[mask_jk], 
                        1/self.LogLike.s2[j,k] * self.Cov[j,k].solve(d_res[j,k,mask_jk])
                    )

                    if plot and rv_i==0.:
                        ax[j].plot(d_wave[j,k], d_res[j,k], c='k', lw=0.8)
                        ax[j].plot(d_wave[j,k], m_res_local_jk, c='C1', lw=1.2)
                        ax[j].set(xlim=(d_wave[j,:].min()-0.3,d_wave[j,:].max()+0.3))
            
        if plot:
            plt.tight_layout()
            plt.show()

        if orders_to_include is None:
            CCF_sum = np.nansum(CCF, axis=(1,2))
        else:
            CCF_sum = CCF[:,orders_to_include,:].sum(axis=(1,2))

        CCF_SNR = convert_CCF_to_SNR(rv, CCF_sum, **kwargs)

        if self.low_memory:
            del self.d_spec, self.Cov, self.LogLike

        return rv, CCF, CCF_SNR

class SpherePlot:

    def _latlon_to_xy(cls, lat, lon, lat_0, lon_0, R=1):

        x = R * np.cos(lat) * np.sin(lon-lon_0)
        y = R * (np.cos(lat_0)*np.sin(lat) - \
                 np.sin(lat_0)*np.cos(lat)*np.cos(lon-lon_0))

        # Check if coordinate is outside observed half-sphere
        c = np.arccos(
            np.sin(lat_0)*np.sin(lat) + \
            np.cos(lat_0)*np.cos(lat)*np.cos(lon-lon_0)
        )

        mask = (c > -np.pi/2) & (c < np.pi/2)
        x[~mask] = np.nan
        y[~mask] = np.nan
        
        return x, y

    def _latlon_to_polar(cls, lat, lon, lat_0, lon_0, R=1):
        
        # Convert to Cartesian? coordinates
        x, y = cls._latlon_to_xy(lat, lon, lat_0, lon_0, R)

        # Convert to polar coordinates
        r   = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        return r, phi

    def __init__(self, Rot, fig, ax, cax=None):

        self.Rot = Rot

        self.fig = fig
        self.ax  = ax
        self.cax = cax

    def plot_map(self, attr, cmap=None, theta_grid_kwargs=None, r_grid_kwargs=None, **kwargs):

        z = getattr(self.Rot, attr)
        if kwargs.get('vmin') is None:
            kwargs['vmin'] = np.nanmin(z)
        if kwargs.get('vmax') is None:
            kwargs['vmax'] = np.nanmax(z)

        for i, r_i in enumerate(self.Rot.unique_r):
            th_i = self.Rot.theta_grid[self.Rot.r_grid==r_i]
            z_i  = z[self.Rot.r_grid==r_i]
            r_i  = np.array([r_i])

            th_i = np.concatenate((th_i-th_i.min(), [2*np.pi]))

            r_i = np.array([
                np.sin(self.Rot.unique_c[i] - (np.pi/2)/self.Rot.n_c/2), 
                np.sin(self.Rot.unique_c[i] + (np.pi/2)/self.Rot.n_c/2*2), 
                ])

            zz_shape = (len(r_i)-1,len(th_i)-1)

            tt, rr = np.meshgrid(th_i, r_i)
            zz = z_i.reshape(zz_shape)

            cntr = self.ax.pcolormesh(
                np.pi/2-tt, rr, zz, shading='auto', cmap=cmap, **kwargs
                )
            
            if theta_grid_kwargs is not None:
                for th_j in th_i:
                    self.ax.plot([np.pi/2-th_j]*2, r_i, **theta_grid_kwargs)
            if r_grid_kwargs is not None:
                self.ax.plot(np.pi/2-th_i, [r_i[0]]*len(th_i), **r_grid_kwargs)
            
    def configure_cax(
            self, label=None, xlim=None, xticks=None, xticklabels=None, vmin=0, vmax=1, 
            cb_width=0.1, scale=1.1, N=256, cmap=None, flip_cb=False, 
            ):

        # Define a grid to plot with pcolormesh
        Z = np.array([np.linspace(0,1+1e-6,N)]*2)
        X = Z.copy()
        if flip_cb:
            X = np.flip(X, axis=-1)
        Y = np.array([np.zeros(N),np.ones(N)])

        self.cax.pcolormesh(
            X, Y, Z, transform=self.cax.transAxes, 
            cmap=cmap, vmin=0, vmax=1, zorder=-1, 
            shading='auto', edgecolors='face', lw=1e-6, 
            )
        
        # Remove ticks/grid
        self.cax.spines['polar'].set_visible(True)
        self.cax.tick_params(top=True, bottom=False)
        self.cax.grid(False)

        self.cax.set(ylim=(0,1), yticks=[])

        # Cut hole in polar axis, creates arch
        r_origin = self.cax.get_ylim()[1] - \
            (self.cax.get_ylim()[1]-self.cax.get_ylim()[0]) / cb_width
        self.cax.set_rorigin(r_origin)

        if xlim is not None:
            # Change the shown angle, rescale and center
            self.set_thetalim(xlim, ax=self.cax, scale=scale)

        if xticks is not None:
            # Add ticks to the colorbar
            theta_min = np.min(self.cax.get_xlim())
            theta_max = np.max(self.cax.get_xlim())

            xticks = np.array(xticks)

            frac   = (xticks-vmin) / (vmax-vmin)
            xticks = xticks[(frac>=0) & (frac<=1)]
            if xticklabels is not None:
                xticklabels = np.array(xticklabels)[(frac>=0) & (frac<=1)]

            frac = frac[(frac>=0) & (frac<=1)]

            if xticklabels is None:
                xticklabels = []
                for xtick_i in xticks:
                    xticklabels.append(str(xtick_i).replace('-',r'$-$'))

            theta_ticks = (theta_max-theta_min)*frac + theta_min
            self.cax.set_xticks(theta_ticks)
            self.cax.set_xticklabels(xticklabels)

        if label is not None:
            # Add a label to the axis
            self.cax.annotate(
                label, xy=(0.5,3.8*scale), #xy=(0.5,3.5*scale), 
                xycoords=self.cax.get_yaxis_transform(), 
                ha='center', va='center'
                )
            
    def configure_ax(
            self, xlim=None, plot_grid=True, grid_lw=0.4, grid_color='k', grid_alpha=1, 
            sep_spine_kwargs={'lw':5, 'c':'w', 'capstyle':'round'}, **kwargs
            ):

        self.ax.grid(False)
        self.ax.set(xticks=[], yticks=[], ylim=(0,1))

        if xlim is not None:
            # Change the shown angle, rescale and center
            self.set_thetalim(xlim)

        if plot_grid:
            # Plot the latitude/longitude grid-lines
            self.grid(
                inc=np.rad2deg(self.Rot.inc), 
                lon_0=np.rad2deg(self.Rot.lon_0), 
                c=grid_color, lw=grid_lw, alpha=grid_alpha
                )
        
        if sep_spine_kwargs is None:
            return
        
        for i, key_i in enumerate(['start', 'end']):
            # Make the spine a separating white line
            self.ax.spines[key_i].set_linewidth(sep_spine_kwargs.get('lw', 5))
            self.ax.spines[key_i].set_color(sep_spine_kwargs.get('c', 'w'))
            self.ax.spines[key_i].set_capstyle(sep_spine_kwargs.get('capstyle', 'round'))
            
            # Plot a black spine around the inner axis edge
            self.ax.axvline(self.ax.get_xlim()[i], lw=sep_spine_kwargs.get('lw', 5)+0.8*2, c='k')

    def grid(
            self, 
            lat_grid=np.arange(-90,90+1e-6,30), 
            lon_grid=np.arange(0,360,30), 
            inc=0, lon_0=0, **kwargs
            ):
        
        # Fine grid of the lines
        lat = np.deg2rad(np.linspace(-90,90,100))
        lon = np.deg2rad(np.linspace(0,360,100))

        inc   = np.deg2rad(inc)
        lon_0 = np.deg2rad(lon_0)

        # Convert to the plot-coordinates and draw
        for lat_i in np.deg2rad(lat_grid):
            r, phi = self._latlon_to_polar(lat_i, lon, inc, lon_0)
            self.ax.plot(phi, r, **kwargs)

        for lon_i in np.deg2rad(lon_grid):
            r, phi = self._latlon_to_polar(lat, lon_i, inc, lon_0)
            self.ax.plot(phi, r, **kwargs)

    def set_thetalim(self, xlim, ax=None, scale=1):

        if ax is None:
            ax = self.ax

        r_origin = ax.get_rorigin()

        # Establish current figure coordinates
        xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        xy_right = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,1))
            )
        width = xy_right[0] - xy_center[0]

        # Change thetalim
        ax.set_thetalim(xlim)

        # Coordinates after thetalim change
        new_xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        new_xy_right = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,1))
            )
        new_width = new_xy_right[0] - new_xy_center[0]

        scale *= width/new_width

        # Rescale the axis
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l, b, w*scale, h*scale])

        # Coordinates after re-scaling
        new_xy_center = self.fig.transFigure.inverted().transform(
            ax.transData.transform((0,r_origin))
            )
        
        xy_offset = xy_center - new_xy_center

        # Recenter the axis
        l, b, w, h = ax.get_position().bounds
        ax.set_position(
            [l+xy_offset[0], b+xy_offset[1], w, h]
            )
        
class CrossCorrPlot:
    
    def __init__(self, fig, ax, cmap='RdBu_r', vsini=0, vtell=0):

        self.fig = fig
        self.ax  = ax

        self.cmap = cmap
        self.vsini = vsini
        self.vtell = vtell

    def colorbar(self, h=0.03, N=50, **kwargs):

        ylim = self.ax.get_ylim()

        # Plot a colorbar at the bottom of the axis
        Z = np.linspace(0,1+1e-6,N).reshape(-1,1).T
        self.ax.imshow(
            Z, extent=[-self.vsini,self.vsini,0,h], origin='lower', 
            aspect='auto', transform=self.ax.get_xaxis_transform(), 
            zorder=-2, cmap=self.cmap, vmin=0, vmax=1, 
            )
        
        self.ax.plot(
            [-self.vsini,-self.vsini,self.vsini,self.vsini], [0,h,h,0], 
            transform=self.ax.get_xaxis_transform(), zorder=-1, **kwargs
            )
        
        self.ax.set(ylim=ylim)
        
    def multicolor(self, rv, CCF, lw, **kwargs):

        points = np.array([rv, CCF]).T[:,None,:]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(vmin=-self.vsini, vmax=self.vsini)
        lc = mpl.collections.LineCollection(
            segments, norm=norm, cmap=self.cmap, capstyle='round', lw=lw
            )
        
        # Set the values used for colormapping
        lc.set_array(rv)
        self.ax.add_collection(lc)

    def plot(
            self, rv, CCF, CCF_other=None, 
            plot_colorbar=False, plot_multicolor=True, 
            **kwargs
            ):

        self.ax.plot(rv, CCF, **kwargs)

        new_kwargs = kwargs.copy()

        if plot_multicolor:
            # Plot a multi-colored line
            rv_mask = (np.abs(rv) < self.vsini)
            new_kwargs['lw'] = kwargs['lw']*4/3
            self.ax.plot(
                rv[rv_mask], CCF[rv_mask], solid_capstyle='round', **new_kwargs
                )
            
            new_kwargs['lw'] = kwargs['lw']*1.7/3           
            self.multicolor(rv[rv_mask], CCF[rv_mask], **new_kwargs)

        if CCF_other is not None:
            new_kwargs['lw']    = kwargs['lw']*1/2
            new_kwargs['alpha'] = 0.3
            self.ax.plot(rv, CCF_other, **new_kwargs, zorder=-1)

        if plot_colorbar:
            # Colorbar at the bottom of the axis
            self.colorbar()

    def configure_ax(
            self, xlabel=r'$v_\mathrm{rad}\ \mathrm{(km\ s^{-1})}$', 
            ylabel=None, xlim=(-120,120), ylim=(-6,6), plot_axvline=True
            ):

        self.ax.set(xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        
        self.ax.spines[['right','top']].set_visible(False)
        self.ax.set_facecolor('none')

        if plot_axvline:
            self.ax.axvline(0, ymin=0, ymax=1, c='k', lw=0.8, zorder=-1)

    def add_xtick_at_vsini(self, length=0.08, return_ticks=False, **kwargs):
        
        ylim = self.ax.get_ylim()
        h = np.abs(ylim[1] - ylim[0])
        
        y = np.array([ylim[0]-h*length/2, ylim[0]+h*length/2])
        if (self.ax.spines['bottom'].get_position() == 'zero'):
            y = np.array([0-h*length/2, 0+h*length/2])

        if return_ticks:
            return y
        
        self.ax.plot([-self.vsini]*2, y, clip_on=False, **kwargs)
        self.ax.plot([+self.vsini]*2, y, clip_on=False, **kwargs)

    def add_xtick_at_vtell(self, length=0.08, **kwargs):

        ylim = self.ax.get_ylim()
        h = np.abs(ylim[1] - ylim[0])
        
        y = np.array([ylim[0]-h*length/2, ylim[0]+h*length/2])
        if (self.ax.spines['bottom'].get_position() == 'zero'):
            y = np.array([0-h*length/2, 0+h*length/2])
        
        self.ax.plot([self.vtell]*2, y, clip_on=False, **kwargs)
