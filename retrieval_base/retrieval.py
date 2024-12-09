import os
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import time

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import pymultinest

from . import utils

class Retrieval:
    """
    A class to handle the retrieval process.

    Attributes:
        config (object): Configuration object containing necessary parameters.
        model_settings (list): List of model settings.
        data_dir (Path): Directory for data output.
        plots_dir (Path): Directory for plots output.
    """

    def __init__(self, config):
        """
        Initialize the Retrieval instance.

        Args:
            config (object): Configuration object containing necessary parameters.
        """
        # Create output directory
        self.config = config
        self.create_output_dir()
        self.model_settings = list(self.config.config_data.keys())

    def create_output_dir(self):
        """
        Create output directories for data and plots, and copy the config file.
        """
        # Create output directory
        self.data_dir = Path(f'{self.config.prefix}data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(f'{self.config.prefix}plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config-file to output directory
        config_file = Path(self.config.file_params)
        destination = self.data_dir / config_file.name
        destination.write_bytes(config_file.read_bytes())

    def save_all_components(self, non_dict_names=[]):
        """
        Save all components to pickle files.

        Args:
            non_dict_names (list): List of non-dictionary attribute names to save.
        """
        for name, component in vars(self).items():

            if name in non_dict_names:
                utils.save_pickle(component, self.data_dir/f'{name}.pkl')

            if not isinstance(component, dict):
                # Skip non-dictionary attributes
                continue
            for m_set, comp in component.items():
                utils.save_pickle(comp, self.data_dir/f'{name}_{m_set}.pkl')

    def load_components(self, component_names):
        """
        Load components from pickle files.

        Args:
            component_names (list): List of component names to load.
        """
        # Pause the process to not overload memory on start-up
        time.sleep(0.3*rank)
        
        for name in component_names:
            component = {}

            # Load the component, if it exists
            file = self.data_dir/f'{name}.pkl'
            if file.exists():
                component = utils.load_pickle(file)

            for m_set in self.model_settings:
                # Load the component, if it exists
                file = self.data_dir/f'{name}_{m_set}.pkl'
                if file.exists():
                    component[m_set] = utils.load_pickle(file)

            if not component:
                # No file found
                continue

            # Make attribute
            setattr(self, name, component)

class RetrievalSetup(Retrieval):
    """
    A class to set up the retrieval.

    Attributes:
        config (object): Configuration object containing necessary parameters.
    """

    def __init__(self, config):
        """
        Initialize the RetrievalSetup instance.

        Args:
            config (object): Configuration object containing necessary parameters.
        """
        # Give arguments to the parent class
        super().__init__(config)
        
        # Pre-process the data or generate synthetic spectrum
        if hasattr(self.config, 'config_data'):
            self.get_data_spectrum(self.config.config_data)
        elif hasattr(self.config, 'config_synthetic_spectrum'):
            self.get_synthetic_spectrum(self.config.config_synthetic_spectrum)
        else:
            raise ValueError('config must have either config_data or config_synthetic_spectrum')

        # Get parameters and model components
        self.get_parameters()
        self.get_model_components()
        self.save_all_components(non_dict_names=['LogLike','Cov','ParamTable'])

    def get_data_spectrum(self, config_data):
        """
        Pre-process the data spectrum.

        Args:
            config_data (dict): Configuration data for the spectrum.
        """
        self.d_spec = {}
        for m_set, config_data_m_set in config_data.items():

            if config_data_m_set['instrument'] == 'CRIRES':
                from .observation import crires as data
            elif config_data_m_set['instrument'] == 'JWST':
                from .observation import jwst as data
            else:
                raise ValueError('Instrument not recognised.')

            # Load and pre-process the target data
            d_spec_target = data.get_class(m_set, config_data_m_set)

            # Summarise in figures and store in dictionary
            d_spec_target.plot_pre_processing(plots_dir=self.plots_dir)
            self.d_spec[m_set] = d_spec_target

    def get_synthetic_spectrum(self, config_synthetic_spectrum, m_set):
        """
        Generate synthetic spectrum. Not implemented.

        Args:
            config_synthetic_spectrum (dict): Configuration for synthetic spectrum.
            m_set (str): Model setting identifier.
        """
        raise NotImplementedError
    
    def get_parameters(self):
        """
        Get parameters for the retrieval process.
        """
        from .parameters import ParameterTable

        # Create a Parameter instance
        self.ParamTable = ParameterTable(
            free_params=self.config.free_params, 
            constant_params=getattr(self.config, 'constant_params', {}), 
            model_settings=self.model_settings, 
            all_model_kwargs=self.config.all_model_kwargs,
            )

    def get_model_components(self):
        """
        Generate model components for the retrieval process.
        """
        self.PT = {}
        self.Chem = {}
        self.Cloud = {}
        self.Rotation = {}

        self.LineOpacity = {}
        self.m_spec = {}

        for m_set in self.model_settings:

            # Query the right model settings
            self.ParamTable.queried_m_set = ['all', m_set]

            # Set the physical model components
            from .model_components import pt_profile
            self.PT[m_set] = pt_profile.get_class(
                **self.ParamTable.PT_kwargs[m_set]
                )
            pressure = self.PT[m_set].pressure
        
            from .model_components import model_spectrum
            self.m_spec[m_set] = model_spectrum.get_class(
                ParamTable=self.ParamTable, 
                d_spec=self.d_spec[m_set], 
                m_set=m_set,
                pressure=pressure, 
                )
            
            from .model_components import line_opacity
            self.LineOpacity[m_set] = line_opacity.get_class(
                m_spec=self.m_spec[m_set],
                line_opacity_kwargs=self.ParamTable.line_opacity_kwargs[m_set], 
                )

            from .model_components import chemistry
            self.Chem[m_set] = chemistry.get_class(
                pressure=pressure, 
                LineOpacity=self.LineOpacity[m_set], 
                **self.ParamTable.chem_kwargs[m_set]
                )

            from .model_components import clouds
            self.Cloud[m_set] = clouds.get_class(
                pressure=pressure, 
                Chem=self.Chem[m_set], 
                PT=self.PT[m_set], 
                **self.ParamTable.cloud_kwargs[m_set]
                )

            from .model_components import rotation_profile
            self.Rotation[m_set] = rotation_profile.get_class(
                **self.ParamTable.rotation_kwargs[m_set]
                )
            
        # Set the observation model components
        sum_model_settings = self.ParamTable.loglike_kwargs.get('sum_model_settings', False)

        from .model_components import covariance
        self.Cov = covariance.get_class(
            d_spec=self.d_spec, 
            sum_model_settings=sum_model_settings, 
            **self.ParamTable.cov_kwargs
            )

        from .model_components import log_likelihood
        self.LogLike = log_likelihood.get_class(
            d_spec=self.d_spec, 
            **self.ParamTable.loglike_kwargs
            )

        self.ParamTable.queried_m_set = 'all'

class RetrievalRun(Retrieval):
    """
    A class to run the retrieval process, including loading data, parameters, and models,
    and initiating the retrieval.

    Attributes:
        config (object): Configuration object containing necessary parameters.
        resume (bool): Flag to indicate if the retrieval should resume from a previous run.
        evaluation (bool): Flag to indicate if it's for evaluation.
        elapsed_times (list): List to store elapsed times for evaluations.
    """

    def __init__(self, config, resume=True, evaluation=False):
        """
        Initialize the RetrievalRun instance.

        Args:
            config (object): Configuration object containing necessary parameters.
            resume (bool): Flag to indicate if the retrieval should resume from a previous run.
            evaluation (bool): Flag to indicate if it's for evaluation.
        """
        # Give arguments to the parent class
        super().__init__(config)

        self.resume     = resume
        self.evaluation = evaluation
        self.elapsed_times = []
        
        # Load a list of components
        component_names = [
            'd_spec', 'ParamTable', 
            'LineOpacity', 'PT', 'Chem', 'Cloud', 'Rotation', 'm_spec', 
            'LogLike', 'Cov'
        ]
        self.load_components(component_names)

    def run_evaluation(self):
        """
        Run the evaluation process.
        """
        # Try loading the evaluation model
        self.load_components(['m_spec_broad', 'LineOpacity_broad'])
        
        if not hasattr(self, 'm_spec_broad') and (rank == 0):
            # Set up the evaluation model (only for master process)
            self.m_spec_broad = {}
            self.LineOpacity_broad = {}

            from .model_components import model_spectrum, line_opacity

            for m_set in self.model_settings:
                self.ParamTable.queried_m_set = ['all', m_set]

                self.m_spec_broad[m_set] = model_spectrum.get_class(
                    ParamTable=self.ParamTable, 
                    d_spec=self.d_spec[m_set], 
                    m_set=m_set,
                    pressure=self.PT[m_set].pressure, 
                    evaluation=self.evaluation
                    )
                utils.save_pickle(
                    self.m_spec_broad[m_set], self.data_dir/f'm_spec_broad_{m_set}.pkl'
                    )
                
                self.LineOpacity_broad[m_set] = line_opacity.get_class(
                    m_spec=self.m_spec_broad[m_set],
                    line_opacity_kwargs=self.ParamTable.line_opacity_kwargs[m_set], 
                    )
                utils.save_pickle(
                    self.LineOpacity_broad[m_set], self.data_dir/f'LineOpacity_broad_{m_set}.pkl'
                    )
                
            self.ParamTable.queried_m_set = 'all'

        # Pause until master process has caught up
        comm.Barrier()

        # Load the evaluation model for all processes
        self.load_components(['m_spec_broad', 'LineOpacity_broad'])

        # Run the callback function
        self.callback(*[None,]*10)
        
    def run(self):
        """
        Run the retrieval process using pymultinest.
        """
        pymultinest.run(
            LogLikelihood=self.get_likelihood, 
            Prior=self.ParamTable, 
            n_dims=self.ParamTable.n_free_params,

            outputfiles_basename=self.config.prefix, 
            dump_callback=self.callback,
            
            resume=self.resume, 
            **self.config.pymultinest_kwargs
            )

    def get_likelihood(self, cube=None, ndim=None, nparams=None, evaluation=False, skip_radtrans=False):
        """
        Calculate the likelihood for the retrieval process.

        Args:
            cube (array): Parameter cube.
            ndim (int): Number of dimensions.
            nparams (int): Number of parameters.
            evaluation (bool): Flag to indicate if it's for evaluation.
            skip_radtrans (bool): Flag to skip radiative transfer calculations.

        Returns:
            float: Log-likelihood value.
        """
        time_start = time.time()
        # ParamTable is updated

        for m_set in self.model_settings:

            self.ParamTable.queried_m_set = ['all', m_set]

            # Update the PT profile
            flag = self.PT[m_set](self.ParamTable)
            if flag == -np.inf:
                # Invalid temperature profile
                return -np.inf
                        
            # Update the chemistry
            self.Chem[m_set](self.ParamTable, temperature=self.PT[m_set].temperature)
            if flag == -np.inf:
                # Invalid chemistry
                return -np.inf
            
            # Update the cloud
            self.Cloud[m_set](
                self.ParamTable, Chem=self.Chem[m_set], PT=self.PT[m_set], 
                mean_wave_micron=np.nanmean(self.d_spec[m_set].wave)*1e-3
                )
            
            if skip_radtrans:
                continue

            # Update the line opacities
            if self.LineOpacity[m_set] is not None:
                for LineOpacity_i in self.LineOpacity[m_set]:
                    LineOpacity_i(self.ParamTable, PT=self.PT[m_set], Chem=self.Chem[m_set])

            # Update the rotation profile
            self.Rotation[m_set](self.ParamTable)

            # Update the model spectrum
            self.m_spec[m_set].evaluation = evaluation
            self.m_spec[m_set](
                self.ParamTable, 
                Chem=self.Chem[m_set], 
                PT=self.PT[m_set], 
                Cloud=self.Cloud[m_set], 
                Rotation=self.Rotation[m_set], 
                LineOpacity=self.LineOpacity[m_set], 
                )
            self.m_spec[m_set].evaluation = False

            if not self.evaluation:
                continue

            # Update the broadened line opacity too
            if self.LineOpacity_broad[m_set] is not None:
                for LineOpacity_i in self.LineOpacity_broad[m_set]:
                    LineOpacity_i(self.ParamTable, PT=self.PT[m_set], Chem=self.Chem[m_set])
            
            # Update the broadened model spectrum too
            self.m_spec_broad[m_set].evaluation = evaluation
            self.m_spec_broad[m_set](
                self.ParamTable, 
                Chem=self.Chem[m_set], 
                PT=self.PT[m_set], 
                Cloud=self.Cloud[m_set], 
                Rotation=self.Rotation[m_set], 
                LineOpacity=self.LineOpacity_broad[m_set], 
                )
            self.m_spec_broad[m_set].evaluation = False
            
        self.ParamTable.queried_m_set = 'all'

        if skip_radtrans:
            return

        sum_model_settings = self.ParamTable.loglike_kwargs.get('sum_model_settings', False)
        m_set_first, *m_set_others = self.model_settings

        other_m_spec = [self.m_spec[m_set] for m_set in m_set_others]
        wave, flux, flux_binned = self.m_spec[m_set_first].combine_model_settings(
            *other_m_spec, sum_model_settings=sum_model_settings
            )
        
        # Update the covariance
        for Cov_i in self.Cov:
            Cov_i(self.ParamTable)

        # Update the log-likelihood
        self.LogLike(m_flux=flux_binned, Cov=self.Cov)

        time_end = time.time()
        self.elapsed_times.append(time_end - time_start)
        
        return self.LogLike.ln_L
    
    def load_posterior_and_bestfit(self, posterior, n_samples_max=2000):
        """
        Load the posterior and best-fit parameters.

        Args:
            posterior (array): Posterior samples.
            n_samples_max (int): Maximum number of samples to load.

        Returns:
            tuple: Posterior samples and best-fit parameters.
        """
        if self.evaluation:
            # Read the equally-weighted posterior
            analyzer = pymultinest.Analyzer(
                n_params=self.ParamTable.n_free_params, 
                outputfiles_basename=self.config.prefix
                )
            posterior = analyzer.get_equal_weighted_posterior()
            posterior = posterior[:,:-1]
            
            # Read best-fit parameters
            stats = analyzer.get_stats()
            bestfit_parameters = np.array(stats['modes'][0]['maximum a posterior'])

        else:
            # Read best-fit parameters
            ln_L = posterior[:,-2]
            bestfit_parameters = posterior[np.argmax(ln_L),:-2]

            # Use only the last n_samples
            n_samples = min(len(posterior), n_samples_max)
            posterior = posterior[-n_samples:,:-2]

        # Evaluate the model with the best-fit parameters
        self.ParamTable.set_apply_prior(False)
        self.ParamTable(cube=bestfit_parameters)
        self.ParamTable.set_apply_prior(True)

        # Update the model components
        self.get_likelihood(evaluation=True, skip_radtrans=False)

        return posterior, bestfit_parameters
    
    def get_profile_posterior(self, posterior):
        """
        Get the posterior profiles for temperature, VMRs, and cloud opacity.

        Args:
            posterior (array): Posterior samples.
        """
        n_samples = posterior.shape[0]

        for m_set in self.model_settings:
            # Number of pressure levels
            n_atm_layers = self.PT[m_set].n_atm_layers

            # Set up posterior definitions
            self.PT[m_set].temperature_posterior = np.nan * np.ones((n_samples, n_atm_layers))
            
            self.Chem[m_set].VMRs_posterior = {
                species_i: np.nan * np.ones((n_samples, n_atm_layers)) \
                for species_i in self.Chem[m_set].VMRs.keys()
                }
            
            if not hasattr(self.Cloud[m_set], 'total_opacity'):
                continue
            self.Cloud[m_set].total_opacity_posterior = np.nan * np.ones((n_samples, n_atm_layers))
    
        for i, sample in enumerate(posterior):
            # Evaluate the model for this sample
            self.ParamTable.set_apply_prior(False)
            self.ParamTable(cube=sample)
            self.ParamTable.set_apply_prior(True)

            # Update the model components, avoid radiative transfer
            self.get_likelihood(evaluation=True, skip_radtrans=True)

            for m_set in self.model_settings:

                # Add to the posterior profiles
                self.PT[m_set].temperature_posterior[i] = self.PT[m_set].temperature

                for species_j, VMR_j in self.Chem[m_set].VMRs.items():
                    self.Chem[m_set].VMRs_posterior[species_j][i] = VMR_j

                if not hasattr(self.Cloud[m_set], 'total_opacity_posterior'):
                    continue
                self.Cloud[m_set].total_opacity_posterior[i] = self.Cloud[m_set].total_opacity

    def callback(self, n_samples, n_live, n_params, live_points, posterior, stats, max_ln_L, ln_Z, ln_Z_err, nullcontext):
        """
        Callback function for pymultinest.

        Args:
            n_samples (int): Number of samples.
            n_live (int): Number of live points.
            n_params (int): Number of parameters.
            live_points (array): Live points.
            posterior (array): Posterior samples.
            stats (dict): Statistics.
            max_ln_L (float): Maximum log-likelihood.
            ln_Z (float): Log-evidence.
            ln_Z_err (float): Log-evidence error.
            nullcontext (object): Null context.
        """
        if (rank != 0):
            # Only the master process should make the live plots
            return
        
        # Update the model components to the best-fit
        posterior, bestfit_parameters = self.load_posterior_and_bestfit(posterior)
        labels = self.ParamTable.get_mathtext()

        # Print the best-fit parameters
        print('\n'+'='*50+'\n')

        mode = 'evaluation' if self.evaluation else 'callback'
        print('Mode: {}'.format(mode))
        print('Time per evaluation: {:.2f} s'.format(np.mean(self.elapsed_times)))
        self.elapsed_times = [] # Reset the timer

        utils.print_bestfit_params(self.ParamTable, self.LogLike)
        print('\n'+'='*50+'\n')

        # Plot best-fitting spectrum
        for m_set in self.model_settings:
            self.d_spec[m_set].plot_bestfit(
                plots_dir=self.plots_dir, 
                LogLike=self.LogLike,
                Cov=self.Cov, 
                )
            if self.LogLike.sum_model_settings:
                break
        
        if self.evaluation:
            # Get the vertical profile posteriors
            self.get_profile_posterior(posterior)

        # Make summarising figure
        from .model_components import figures_model
        figures_model.plot_summary(
            plots_dir=self.plots_dir,
            posterior=posterior, 
            bestfit_parameters=bestfit_parameters, 
            labels=labels, 
            PT=self.PT, 
            Chem=self.Chem, 
            Cloud=self.Cloud, 
            m_spec=self.m_spec, 
            evaluation=self.evaluation,
            )

        if self.evaluation:
            # Save best-fitting model components
            self.save_all_components(non_dict_names=['LogLike','Cov','ParamTable'])