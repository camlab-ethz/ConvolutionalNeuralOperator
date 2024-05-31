from DataLoaders.CNO_TimeLoaders import BrownianBridgeTimeDataset, VortexSheetTimeDataset, SinesTimeDataset, PiecewiseConstantsTimeDataset, GaussiansTimeDataset, ComplicatedShearLayerTimeDataset, KelvinHelmholtzTimeDataset, RiemannTimeDataset, RiemannCurvedTimeDataset, EulerGaussTimeDataset, RiemannKHTimeDataset, RichtmyerMeshkov, RayleighTaylor, PoissonGaussians,  Helmholtz, AllenCahn, WaveSeismic, WaveGaussians,PiecewiseConstantsTraceTimeDataset, KolmogorovFlow, Airfoil

def _load_dataset(dic, 
                  which, 
                  which_loader,
                  in_dim,
                  out_dim,
                  masked_input = None,
                  fix_input_to_time_step = None):
    
    num_samples = dic["num_samples"]
    if "ns_" in which and dic["num_samples"]>19640:
        num_samples = 19640
    if "eul_" in which and dic["num_samples"]>9640:
        num_samples = 9640
    
    print("WHICH: ", which, " NUM_SAMPLES: ", num_samples)


    if which == "ns_brownian":
            
        train_dataset =   BrownianBridgeTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                    time_step_size = dic["dt"],
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_loader,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = num_samples,
                                                    data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                    time_input = dic["time_input"],
                                                    masked_input = masked_input,
                                                    allowed_transitions = dic["allowed_tran"])
    elif which == "ns_pwc":
        train_dataset =   PiecewiseConstantsTimeDataset(
                                            max_num_time_steps = dic["time_steps"], 
                                            time_step_size = dic["dt"],
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_loader,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = num_samples,
                                            data_path = "---- PROVIDE THE FOLDER PATH ----",
                                            time_input = dic["time_input"],
                                            masked_input = masked_input,
                                            allowed_transitions = dic["allowed_tran"])
    elif which == "ns_gauss":
        train_dataset  =   GaussiansTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                time_step_size = dic["dt"],
                                                fix_input_to_time_step = fix_input_to_time_step,
                                                which = which_loader,
                                                resolution = 128,
                                                in_dist = True,
                                                num_trajectories = num_samples,
                                                data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                time_input = dic["time_input"],
                                                masked_input = masked_input,
                                                allowed_transitions = dic["allowed_tran"],)
    elif which == "ns_sin":
        train_dataset  =   SinesTimeDataset(max_num_time_steps = dic["time_steps"], 
                                            time_step_size = dic["dt"],
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_loader,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = num_samples,
                                            data_path ="---- PROVIDE THE FOLDER PATH ----",
                                            time_input = dic["time_input"],
                                            masked_input = masked_input,
                                            allowed_transitions = dic["allowed_tran"])

    elif which == "ns_vortex":
        train_dataset  = VortexSheetTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                time_step_size = dic["dt"],
                                                fix_input_to_time_step = fix_input_to_time_step,
                                                which = which_loader,
                                                resolution = 128,
                                                in_dist = True,
                                                num_trajectories = num_samples,
                                                data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                time_input = dic["time_input"],
                                                masked_input = masked_input,
                                                allowed_transitions = dic["allowed_tran"])
    elif which == "ns_shear":
        train_dataset = ComplicatedShearLayerTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                        time_step_size = dic["dt"],
                                                        fix_input_to_time_step = fix_input_to_time_step,
                                                        which = which_loader,
                                                        resolution = 128,
                                                        in_dist = True,
                                                        num_trajectories = num_samples,
                                                        data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                        time_input = dic["time_input"],
                                                        masked_input = masked_input,
                                                        allowed_transitions = dic["allowed_tran"])

    elif which == "ns_pwc_t":
        train_dataset =   PiecewiseConstantsTraceTimeDataset(
                                            max_num_time_steps = dic["time_steps"], 
                                            time_step_size = dic["dt"],
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_loader,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = num_samples,
                                            data_path = "---- PROVIDE THE FOLDER PATH ----",
                                            time_input = dic["time_input"],
                                            masked_input = masked_input,
                                            allowed_transitions = dic["allowed_tran"])

    elif which == "eul_kh":
        train_dataset =   KelvinHelmholtzTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                    time_step_size = dic["dt"],
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_loader,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = num_samples,
                                                    data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                    time_input = dic["time_input"],
                                                    masked_input = masked_input, 
                                                    allowed_transitions = dic["allowed_tran"])
    elif which == "eul_riemann":
        train_dataset =   RiemannTimeDataset(max_num_time_steps = dic["time_steps"], 
                                            time_step_size = dic["dt"],
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_loader,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = num_samples,
                                            data_path = "---- PROVIDE THE FOLDER PATH ----",
                                            time_input = dic["time_input"],
                                            masked_input = masked_input,
                                            allowed_transitions = dic["allowed_tran"])
    elif which == "eul_riemann_cur":
        train_dataset =   RiemannCurvedTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                time_step_size = dic["dt"],
                                                fix_input_to_time_step = fix_input_to_time_step,
                                                which = which_loader,
                                                resolution = 128,
                                                in_dist = True,
                                                num_trajectories = num_samples,
                                                data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                time_input = dic["time_input"],
                                                masked_input = masked_input,
                                                allowed_transitions = dic["allowed_tran"])
    elif which == "eul_gauss":
        train_dataset =   EulerGaussTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                time_step_size = dic["dt"],
                                                fix_input_to_time_step = fix_input_to_time_step,
                                                which = which_loader,
                                                resolution = 128,
                                                in_dist = True,
                                                num_trajectories = num_samples,
                                                data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                time_input = dic["time_input"],
                                                masked_input = masked_input,
                                                allowed_transitions = dic["allowed_tran"])
    elif which == "eul_riemann_kh":
        train_dataset =   RiemannKHTimeDataset(max_num_time_steps = dic["time_steps"], 
                                                time_step_size = dic["dt"],
                                                fix_input_to_time_step = fix_input_to_time_step,
                                                which = which_loader,
                                                resolution = 128,
                                                in_dist = True,
                                                num_trajectories = num_samples,
                                                data_path = "---- PROVIDE THE FOLDER PATH ----",
                                                time_input = dic["time_input"],
                                                masked_input = masked_input,
                                                allowed_transitions = dic["allowed_tran"])
    
    elif which == "rich_mesh":
        train_dataset =   RichtmyerMeshkov(max_num_time_steps = dic["time_steps"], 
                                            time_step_size = dic["dt"],
                                            fix_input_to_time_step = fix_input_to_time_step,
                                            which = which_loader,
                                            resolution = 128,
                                            in_dist = True,
                                            num_trajectories = num_samples,
                                            data_path = "---- PROVIDE THE FOLDER PATH ----",
                                            time_input = dic["time_input"],
                                            masked_input = masked_input,
                                            allowed_transitions = dic["allowed_tran"],
                                            tracer = False)

        
    elif which == "rayl_tayl":
        train_dataset =   RayleighTaylor(max_num_time_steps = dic["time_steps"], 
                                        time_step_size = dic["dt"],
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_loader,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = num_samples,
                                        data_path = "---- PROVIDE THE FOLDER PATH ----",
                                        time_input = dic["time_input"],
                                        masked_input = masked_input,
                                        allowed_transitions = dic["allowed_tran"],
                                        tracer = False)
            
    elif which == "kolmogorov":
        train_dataset =   KolmogorovFlow(max_num_time_steps = dic["time_steps"], 
                                        time_step_size = dic["dt"],
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_loader,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = num_samples,
                                        data_path = "---- PROVIDE THE FOLDER PATH ----",
                                        time_input = dic["time_input"],
                                        masked_input = masked_input,
                                        allowed_transitions = dic["allowed_tran"])
        
    elif which == "airfoil":
        
        train_dataset = Airfoil(which = which_loader,
                               resolution = 128,
                               in_dist = True,
                               num_trajectories = num_samples,
                               data_path = "---- PROVIDE THE FOLDER PATH ----",
                               time_input = False,
                               masked_input = None)
    
    elif which == "allen_cahn":
        train_dataset =   AllenCahn(max_num_time_steps = dic["time_steps"], 
                                    time_step_size = dic["dt"],
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_loader,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = num_samples,
                                    data_path = "---- PROVIDE THE FOLDER PATH ----",
                                    time_input = dic["time_input"],
                                    masked_input = masked_input,
                                    allowed_transitions = dic["allowed_tran"])
    
    elif which == "wave_seismic":
        train_dataset =   WaveSeismic(max_num_time_steps = dic["time_steps"], 
                                    time_step_size = dic["dt"],
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_loader,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = num_samples,
                                    data_path = "---- PROVIDE THE FOLDER PATH ----",
                                    time_input = dic["time_input"],
                                    masked_input = masked_input,
                                    allowed_transitions = dic["allowed_tran"])
    
    elif which == "wave_gauss":
        train_dataset =   WaveGaussians(max_num_time_steps = dic["time_steps"], 
                                        time_step_size = dic["dt"],
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_loader,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = num_samples,
                                        data_path = "---- PROVIDE THE FOLDER PATH ----",
                                        time_input = dic["time_input"],
                                        masked_input = masked_input,
                                        allowed_transitions = dic["allowed_tran"])

    elif which == "poisson_gauss":
        train_dataset =   PoissonGaussians(which = which_loader,
                                           resolution = 128,
                                           in_dist = True,
                                           num_trajectories = num_samples,
                                           data_path = "---- PROVIDE THE FOLDER PATH ----",
                                           time_input = False,
                                           masked_input = None)
        
    elif which == "helmholtz":
        train_dataset =   Helmholtz(which = which_loader,
                                   resolution = 128,
                                   in_dist = True,
                                   num_trajectories = num_samples,
                                   data_path = "---- PROVIDE THE FOLDER PATH ----",
                                   time_input = False,
                                   masked_input = None)
    
    
    else:
        raise ValueError("Not implemented experiment")

    return train_dataset