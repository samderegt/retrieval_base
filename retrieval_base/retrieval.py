from pathlib import Path
import time
import warnings

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
        self._create_output_dir()
        self.model_settings = list(self.config.config_data.keys())

    def save_all_components(self, non_dict_names=[]):
        """
        Save all components to pickle files.

        Args:
            non_dict_names (list): List of non-dictionary attribute names to save.
        """
        for name, component in vars(self).items():

            if name in non_dict_names:
                utils.save_pickle(component, self.data_dir/f'{name}.pkl')
                continue

            if not isinstance(component, dict):
                # Skip non-dictionary attributes
                warnings.warn(f'Attribute {name} is not a dictionary, not saving.')
                continue

            # Is a dictionary, loop over model settings
            for m_set, comp in component.items():
                # Save the component
                utils.save_pickle(comp, self.data_dir/f'{name}_{m_set}.pkl')

                if getattr(comp, 'shared_between_m_set', False):
                    # Only save the shared component once
                    break

    def load_components(self, component_names):
        """
        Load components from pickle files.

        Args:
            component_names (list): List of component names to load.
        """
        
        for name in component_names:
            shared_between_m_set = False
            component = {}

            # Load the component, if it exists
            file = self.data_dir/f'{name}.pkl'
            if file.exists():
                # Not specific to a model setting
                component = utils.load_pickle(file)

            for m_set in self.model_settings:

                if shared_between_m_set and (m_set != self.model_settings[0]):
                    # Refer to the first model setting
                    component[m_set] = component[self.model_settings[0]]

                # Load the setting-specific component
                file = self.data_dir/f'{name}_{m_set}.pkl'
                if file.exists():
                    component[m_set] = utils.load_pickle(file)
                    
                    # Check if the component is shared between model settings
                    shared_between_m_set = getattr(component[m_set], 'shared_between_m_set', False)

            if not component:
                continue # No file found

            # Make attribute
            setattr(self, name, component)
    
    def _create_output_dir(self):
        """
        Create output directories for data and plots, and copy the config file.
        """
        # Create output directory
        self.data_dir = Path(f'{self.config.prefix}data')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path(f'{self.config.prefix}plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config-file to output directory
        config_file = Path(self.config.config_file)
        destination = self.data_dir / config_file.name
        destination.write_bytes(config_file.read_bytes())

class RetrievalSetup(Retrieval):
    """
    A class to set up the retrieval.

    Attributes:
        config (object): Configuration object containing necessary parameters.
    """

    def _read_component_from_module(self, name, module, m_set, **kwargs):
        """
        Retrieves or loads a model component from a specified module and updates the component in the instance.

        Args:
            name (str): The name of the model component attribute.
            module (module): The module from which to load the model component.
            m_set (str): The model setting identifier.
            **kwargs: Additional keyword arguments, including:
            shared_between_m_set (bool): If True, the model component is shared between model settings.
        """
        
        # Read the model component if it exists
        component = getattr(self, name, {})

        # Check if model component is shared and not first model setting
        shared_between_m_set = kwargs.get('shared_between_m_set', False)

        if shared_between_m_set and (m_set != self.model_settings[0]):
            # Refer to the first model setting
            component[m_set] = component[self.model_settings[0]]
        else:
            # Load the model component from the module
            component[m_set] = module.get_class(m_set=m_set, **kwargs)
        
        if component[m_set] is not None:
            # Update the shared flag
            component[m_set].shared_between_m_set = shared_between_m_set

        # Update the model component
        setattr(self, name, component)

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
            self.setup_data_spectrum(self.config.config_data)
        elif hasattr(self.config, 'config_synthetic_spectrum'):
            self.setup_synthetic_spectrum(self.config.config_synthetic_spectrum)
        else:
            raise ValueError('config must have either config_data or config_synthetic_spectrum')

        # Get parameters and model components
        self.setup_parameters()
        self.setup_model_components()
        self.save_all_components(non_dict_names=['LogLike','Cov','ParamTable'])

    def setup_data_spectrum(self, config_data):
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

    def setup_synthetic_spectrum(self, config_synthetic_spectrum, m_set):
        """
        Generate synthetic spectrum. Not implemented.

        Args:
            config_synthetic_spectrum (dict): Configuration for synthetic spectrum.
            m_set (str): Model setting identifier.
        """
        raise NotImplementedError
    
    def setup_parameters(self):
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
        
    def setup_model_components(self):
        """
        Generate model components for the retrieval.
        """
        for m_set in self.model_settings:
            self.ParamTable.set_queried_m_set(['all',m_set])
            self._setup_physical_model_components(m_set)

        self.ParamTable.set_queried_m_set('all')

        self._setup_observation_model_components()

    def _setup_physical_model_components(self, m_set):
        """
        Set the physical model components for a given model setting.

        Args:
            m_set (str): Model setting identifier.
        """
        from .model_components import pt_profile, model_spectrum, line_opacity, chemistry, clouds, rotation_profile

        self._read_component_from_module(
            name='PT', module=pt_profile, m_set=m_set, 
            **self.ParamTable.PT_kwargs[m_set]
        )
        pressure = self.PT[m_set].pressure

        self._read_component_from_module(
            name='m_spec', module=model_spectrum, m_set=m_set, 
            ParamTable=self.ParamTable, 
            d_spec=self.d_spec[m_set], 
            pressure=pressure
        )

        self._read_component_from_module(
            name='LineOpacity', module=line_opacity, m_set=m_set, 
            m_spec=self.m_spec[m_set],
            line_opacity_kwargs=self.ParamTable.line_opacity_kwargs[m_set]
        )

        self._read_component_from_module(
            name='Chem', module=chemistry, m_set=m_set, 
            pressure=pressure, 
            LineOpacity=self.LineOpacity[m_set], 
            **self.ParamTable.chem_kwargs[m_set]
        )

        self._read_component_from_module(
            name='Cloud', module=clouds, m_set=m_set, 
            pressure=pressure, 
            Chem=self.Chem[m_set], 
            PT=self.PT[m_set], 
            **self.ParamTable.cloud_kwargs[m_set]
        )

        self._read_component_from_module(
            name='Rotation', module=rotation_profile, m_set=m_set, 
            **self.ParamTable.rotation_kwargs[m_set]
        )

    def _setup_observation_model_components(self):
        """
        Set the observation model components.
        """
        from .model_components import covariance, log_likelihood

        sum_model_settings = self.ParamTable.loglike_kwargs.get('sum_model_settings', False)

        self.Cov = covariance.get_class(
            d_spec=self.d_spec, 
            sum_model_settings=sum_model_settings, 
            **self.ParamTable.cov_kwargs
        )

        self.LogLike = log_likelihood.get_class(
            d_spec=self.d_spec, 
            **self.ParamTable.loglike_kwargs
        )

        self.ParamTable.set_queried_m_set('all')

    def setup_evaluation_model_components(self):
        """
        Generate the (broad) evaluation model components for the retrieval.
        """
        from .model_components import model_spectrum, line_opacity

        for m_set in self.model_settings:
            self.ParamTable.set_queried_m_set(['all',m_set])

            # Generate broadened model spectrum and save as pickle
            self._read_component_from_module(
                name='m_spec_broad', module=model_spectrum, m_set=m_set, 
                ParamTable=self.ParamTable, 
                d_spec=self.d_spec[m_set], 
                pressure=self.PT[m_set].pressure, 
                evaluation=self.evaluation
                )
            utils.save_pickle(self.m_spec_broad[m_set], self.data_dir/f'm_spec_broad_{m_set}.pkl')

            # Generate broadened line opacities and save as pickle
            self._read_component_from_module(
                name='LineOpacity_broad', module=line_opacity, m_set=m_set, 
                m_spec=self.m_spec_broad[m_set],
                line_opacity_kwargs=self.ParamTable.line_opacity_kwargs[m_set], 
                )
            shared_between_m_set = getattr(self.LineOpacity_broad[m_set], 'shared_between_m_set', False)
            if shared_between_m_set and (m_set != self.model_settings[0]):
                # Not the first of shared model settings, don't save
                continue
            
            # Save the component
            utils.save_pickle(self.LineOpacity_broad[m_set], self.data_dir/f'LineOpacity_broad_{m_set}.pkl')
            
        self.ParamTable.set_queried_m_set('all')

class RetrievalRun(RetrievalSetup, Retrieval):
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
        # Pause the process to not overload memory on start-up
        time.sleep(0.3*rank)

        # Give arguments to the parent class
        Retrieval.__init__(self, config)

        self.resume     = resume
        self.evaluation = evaluation
        self.elapsed_times = []
        
        # Load the data, parameter and model components
        self._load_initial_components()

    def run(self):
        """
        Run the retrieval using pymultinest.
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
        
    def run_evaluation(self):
        """
        Run the evaluation.
        """
        # Load the evaluation model components
        self._load_evaluation_components()

        # Run the callback function
        self.callback(*[None,]*10)

    def run_profiling(self, N=50):
        """
        Run the profiling.

        Args:
            N (int): Number of loops for profiling.
        """
        def func():
            np.random.seed(345610)

            for _ in range(N):
                # Create a sample and evaluate the model
                sample = np.random.uniform(0, 1, self.ParamTable.n_free_params)
                self.ParamTable(cube=sample)
                self.get_likelihood(evaluation=False, skip_radtrans=False)
        
        import cProfile, pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        func()
        profiler.disable()

        profiler.dump_stats('profile_output')
        p = pstats.Stats('profile_output')
        p.sort_stats('cumulative').print_stats(25)
        
    def get_likelihood(self, cube=None, ndim=None, nparams=None, evaluation=False, skip_radtrans=False):
        """
        Calculate the likelihood for the retrieval.

        Args:
            cube (array): Parameter cube.
            ndim (int): Number of dimensions.
            nparams (int): Number of parameters.
            evaluation (bool): Flag to indicate if it's for evaluation.
            skip_radtrans (bool): Flag to skip radiative transfer calculations.

        Returns:
            float: Log-likelihood value.
        """

        if not self.ParamTable.is_physical:
            # Reject unphysical parameter values
            return -np.inf

        # ParamTable is already updated by MultiNest
        time_start = time.time()
        
        for m_set in self.model_settings:
            self.ParamTable.set_queried_m_set(['all',m_set])
            
            # Update all model components
            if self._update_pt_profile(m_set) == -np.inf:
                return -np.inf
            if self._update_chemistry(m_set) == -np.inf:
                return -np.inf
            self._update_cloud(m_set)
                        
            if skip_radtrans:
                continue # Skip radiative transfer calculations

            self._update_rotation(m_set)
            self._update_line_opacities(m_set)
            self._update_model_spectrum(m_set, evaluation)
            
        self.ParamTable.set_queried_m_set('all')

        if skip_radtrans:
            return # Skip radiative transfer calculations

        # Combine the spectra of multiple model settings
        flux_binned = self._combine_model_spectra()
        
        # Update the covariance
        for Cov_i in self.Cov:
            Cov_i(self.ParamTable)

        # Compute the log-likelihood
        self.LogLike(m_flux=flux_binned, Cov=self.Cov)

        time_end = time.time()
        self.elapsed_times.append(time_end - time_start)
        
        return self.LogLike.ln_L

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

    def _load_initial_components(self):
        """
        Load initial components from pickle files.
        """
        component_names = [
            'd_spec', 'ParamTable', 
            'LineOpacity', 'PT', 'Chem', 'Cloud', 'Rotation', 'm_spec', 
            'LogLike', 'Cov'
        ]
        self.load_components(component_names)

    def _load_evaluation_components(self):
        """
        Load evaluation components from pickle files.
        """
        # Try loading the evaluation model
        self.load_components(['m_spec_broad', 'LineOpacity_broad'])
        
        if not hasattr(self, 'm_spec_broad') and (rank == 0):
            # Set up the evaluation model (only for master process)
            self.setup_evaluation_model_components()

        # Pause until master process has caught up
        comm.Barrier()

        # Load the evaluation model for all processes
        self.load_components(['m_spec_broad', 'LineOpacity_broad'])

    def _skip_update(self, m_set, component):
        """
        Skip updating a model component.

        Args:
            m_set (str): Model setting identifier.
            component: Model component.

        Returns:
            bool: Flag to skip updating the component.
        """    
        if component[m_set] is None:
            # Always skip if no component exists
            return True    
        if m_set == self.model_settings[0]:
            # Always update the first model setting
            return False
        
        # Skip if the component is shared between model settings
        shared_between_m_set = getattr(component[m_set], 'shared_between_m_set', False)
        return shared_between_m_set

    def _update_pt_profile(self, m_set):
        """
        Update the PT profile for a given model setting.

        Args:
            m_set (str): Model setting identifier.

        Returns:
            float: Flag indicating success or failure.
        """
        if self._skip_update(m_set, self.PT):
            return
        return self.PT[m_set](self.ParamTable)

    def _update_chemistry(self, m_set):
        """
        Update the chemistry for a given model setting.

        Args:
            m_set (str): Model setting identifier.

        Returns:
            float: Flag indicating success or failure.
        """
        if self._skip_update(m_set, self.Chem):
            return
        return self.Chem[m_set](self.ParamTable, temperature=self.PT[m_set].temperature)

    def _update_cloud(self, m_set):
        """
        Update the cloud for a given model setting.

        Args:
            m_set (str): Model setting identifier.
        """
        if self._skip_update(m_set, self.Cloud):
            return
        self.Cloud[m_set](
            self.ParamTable, Chem=self.Chem[m_set], PT=self.PT[m_set], 
            mean_wave_micron=np.nanmean(self.d_spec[m_set].wave)*1e-3
            )

    def _update_rotation(self, m_set):
        """
        Update the rotation profile for a given model setting.

        Args:
            m_set (str): Model setting identifier.
        """
        if self._skip_update(m_set, self.Rotation):
            return
        self.Rotation[m_set](self.ParamTable)

    def _update_line_opacities(self, m_set):
        """
        Update the line opacities for a given model setting.

        Args:
            m_set (str): Model setting identifier.
        """
        if self._skip_update(m_set, self.LineOpacity):
            return
        for LineOpacity_i in self.LineOpacity[m_set]:
            LineOpacity_i(self.ParamTable, PT=self.PT[m_set], Chem=self.Chem[m_set])

    def _update_model_spectrum(self, m_set, evaluation):
        """
        Update the model spectrum for a given model setting.

        Args:
            m_set (str): Model setting identifier.
            evaluation (bool): Flag to indicate if it's for evaluation.
        """
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

        if self.evaluation:
            # Update the broadened model spectrum too

            if not self._skip_update(m_set, self.LineOpacity_broad):
                for LineOpacity_i in self.LineOpacity_broad[m_set]:
                    LineOpacity_i(self.ParamTable, PT=self.PT[m_set], Chem=self.Chem[m_set])
            
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

    def _combine_model_spectra(self):
        """
        Combine the model spectra.
        """
        sum_model_settings = self.ParamTable.loglike_kwargs.get('sum_model_settings', False)
        m_set_first, *m_set_others = self.model_settings

        other_m_spec = [self.m_spec[m_set] for m_set in m_set_others]
        wave, flux, flux_binned = self.m_spec[m_set_first].combine_model_settings(
            *other_m_spec, sum_model_settings=sum_model_settings
            )
        return flux_binned