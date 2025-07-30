import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import os
import pandas as pd
import numpy as np
import itertools
import glob
import datetime
from filelock import FileLock



#################### Delphos ####################
base = importr('base')
apollo = importr('apollo')

def apollo_probabilities(apollo_beta, apollo_inputs, attributes, covariates, apollo_beta_estimated, r_print=False):
    
    num_alternatives = max(attributes.keys())

    ro.globalenv['apollo_beta'] = apollo_beta
    ro.globalenv['apollo_inputs'] = apollo_inputs
    ro.globalenv['apollo_estimated'] = apollo_beta_estimated

    utility_functions = []    
    for alt in range(1, num_alternatives + 1):
            terms = []
                
            # Alternative-Specific Constants
            asc_param = f'asc_alt{alt}'
            if asc_param in apollo_beta_estimated:
                terms.append(asc_param)   
            
            for cov in covariates:
                min_category = min(covariates[cov])
                max_category = max(covariates[cov])                
                asc_cov_param = f'asc_alt{alt}_{cov}_{min_category}'
                if asc_cov_param in apollo_beta_estimated:
                    interaction_terms = [f'asc_alt{alt}_{cov}_{k} * ({cov} == {k})' for k in range(min_category, max_category + 1)]
                    terms.append(f'({" + ".join(interaction_terms)})')

            # Attribute-specific Coefficients
            if alt in attributes:
                for attr in attributes[alt]:                    
                    # Generic terms
                    generic_terms = [
                        (f'b_{attr}_generic', f'x_{alt}_{attr}'),
                        (f'b_{attr}_generic_log', f'log(x_{alt}_{attr})'),
                        (f'b_{attr}_generic_box_cox', f'(x_{alt}_{attr}**L_{attr} - 1)/L_{attr}')
                    ]
                    
                    for param, expr in generic_terms:
                        if param in apollo_beta_estimated:
                            terms.append(f'({param} * {expr})')

                    # Alternative-specific terms
                    alt_specific_terms = [
                        (f'b_{alt}_{attr}', f'x_{alt}_{attr}'),
                        (f'b_{alt}_{attr}_log', f'log(x_{alt}_{attr})'),
                        (f'b_{alt}_{attr}_box_cox', f'(x_{alt}_{attr}**L_{alt}_{attr} - 1)/L_{alt}_{attr}')
                    ]
                    for param, expr in alt_specific_terms:
                        if param in apollo_beta_estimated:
                            terms.append(f'({param} * {expr})')

                    # Interaction terms
                    for cov in covariates:
                        min_category = min(covariates[cov])
                        max_category = max(covariates[cov])  

                        generic_cov_terms = [(f'b_{attr}_generic_{cov}_{k}', f'x_{alt}_{attr} * ({cov} == {k})') for k in range(min_category, max_category + 1)]
                        generic_terms_to_add = []
                        for param, expr in generic_cov_terms:
                            if param in apollo_beta_estimated:
                                generic_terms_to_add.append(f"({param} * {expr})")
                        if generic_terms_to_add:
                            terms.append(f'({" + ".join(generic_terms_to_add)})')
                        
                        alt_cov_terms = [(f'b_{alt}_{attr}_{cov}_{k}', f'x_{alt}_{attr} * ({cov} == {k})') for k in range(min_category, max_category + 1)]
                        alt_terms_to_add = []
                        for param, expr in alt_cov_terms:
                            if param in apollo_beta_estimated:                    
                                alt_terms_to_add.append(f"({param} * {expr})")
                        if alt_terms_to_add:
                            terms.append(f'({" + ".join(alt_terms_to_add)})')
            if not terms:
                terms.append('0')

            utility_functions.append(f'V[["Alt{alt}"]] = ' + " + ".join(terms))
    

    utility_code = "\n        ".join(utility_functions)

    r_code = f"""
         apollo_probabilities_function <- function(apollo_beta, apollo_inputs, functionality = 'estimate') {{
            apollo_attach(apollo_beta, apollo_inputs)
            on.exit(apollo_detach(apollo_beta, apollo_inputs))
            
            P = list()
            V = list()
            
            {utility_code}
            
            mnl_settings = list(
                alternatives = c({", ".join([f"Alt{alt}={alt}" for alt in range(1, num_alternatives + 1)])}),
                avail = list({", ".join([f"Alt{alt}=avail_{alt}" for alt in range(1, num_alternatives + 1)])}),
                choiceVar = choice,
                utilities = V
            )
            
            P[["model"]] = apollo_mnl(mnl_settings, functionality)
            P = apollo_panelProd(P, apollo_inputs, functionality)
            P = apollo_prepareProb(P, apollo_inputs, functionality)
            return(P)
        }}
        """
    if r_print:
        print(r_code)
    else:
        ro.r(r_code)
        return ro.globalenv['apollo_probabilities_function']

def mnl_interaction(name, apollo_beta_fixed, apollo_beta_estimated, attributes, covariates, job_index, path_dir='specification', df = 'train.csv', info = False):

    """
    Dynamically configures and runs an MNL interaction model using Apollo.

    Args:
        name: Name of the model.
        case: Specific case or directory name.
        beta_fixed: Fixed beta parameters for the model.
        num_alternatives: Number of alternatives in the choice model.
        num_attributes: Number of attributes per alternative.
        path_dir: Directory for model files and outputs.
    """
    num_alternatives = max(attributes.keys())
    num_attributes = max(max(attr_list) for attr_list in attributes.values())

    # Directories
    output_directory   = f'{path_dir}/outputs/job_{job_index}'
    database_directory = f'{path_dir}/{df}'    
    save_directory     = f'{path_dir}/outputs/job_{job_index}/{name}_results.csv'
    
    apollo.apollo_initialise()

    ro.globalenv['apollo_control']  = ro.ListVector({'modelName': f"{name}",  
                                                     'modelDescr': "MNL model on choice SP data", 
                                                     'indivID': "ID", 
                                                     'outputDirectory': f"{output_directory}"})
    
    
    ###### In sample and out of sample data ######
    ro.globalenv['database_1'] = ro.r(f'read.csv("{database_directory}", header=TRUE, sep = ",")')    

    ro.r('''    
        set.seed(123) 
        individuals <- unique(database_1$ID)
        n_individuals <- length(individuals)

        ### Split into 80% train and 20% test
        train_individuals <- sample(individuals, size = 0.8 * n_individuals)
        test_individuals  <- setdiff(individuals, train_individuals)
        database_1$test <- ifelse(database_1$ID %in% test_individuals, 1, 0)
        in_sample <- subset(database_1, test == 0)
         
        # We are going to estimate in_sample / out_of_sample metrics 
        database  <- database_1                  
        ''')
 
    beta_values = []
    beta_names = []

    # Add alternative-specific constants
    for alt in range(1, num_alternatives + 1):
        beta_values.append(0)
        beta_names.append(f'asc_alt{alt}')
        for cov in covariates:
            for category in covariates[cov]:
                beta_values.append(0)
                beta_names.append(f'asc_alt{alt}_{cov}_{category}')

    # Add generic coefficients and transformations for attributes
    for attr in range(1, num_attributes + 1):
        # Generic taste parameters (for all alternatives)
        beta_values.extend([0, 0, 0, 0.1])  
        beta_names.extend([f'b_{attr}_generic', f'b_{attr}_generic_log', f'b_{attr}_generic_box_cox', f'L_{attr}'])

        for cov in covariates:
            for category in covariates[cov]:  
                beta_values.extend([0, 0, 0])  
                beta_names.extend([f'b_{attr}_generic_{cov}_{category}', f'b_{attr}_generic_log_{cov}_{category}', f'b_{attr}_generic_box_cox_{cov}_{category}'])

        # Specific taste parameters (for each alternative)
        for alt in range(1, num_alternatives + 1):
            beta_values.extend([0, 0, 0, 0.1])  # 0 for coefficients, 0.1 for Box-Cox parameter (L)
            beta_names.extend([f'b_{alt}_{attr}', f'b_{alt}_{attr}_log', f'b_{alt}_{attr}_box_cox', f'L_{alt}_{attr}'])

            for cov in covariates:
                for category in covariates[cov]:  # Iterate over each category
                    beta_values.extend([0, 0, 0])  
                    beta_names.extend([f'b_{alt}_{attr}_{cov}_{category}', f'b_{alt}_{attr}_log_{cov}_{category}', f'b_{alt}_{attr}_box_cox_{cov}_{category}'])

    
   ############# Apollo functions   #############
        
    apollo_beta                         = ro.FloatVector(beta_values)
    apollo_beta.names                   = ro.StrVector(beta_names)

    ro.globalenv['apollo_beta']         = apollo_beta
    ro.globalenv['apollo_fixed']        = ro.StrVector(apollo_beta_fixed)

    apollo_inputs                       = apollo.apollo_validateInputs()
    apollo_probs                        = apollo_probabilities(apollo_beta, apollo_inputs, attributes, covariates, apollo_beta_estimated, False)    
    estimate_sett                       = ro.ListVector({'printLevel': 0,'writeIter': False,'silent': True})
    ro.globalenv['estimate_settings']   = estimate_sett
    
    #outOfSample_settings                =  ro.r('''list(samples = cbind(database_1$test, database_1$test), rmse = NULL)''')
    #apollo.apollo_outOfSample(apollo_beta, ro.globalenv['apollo_fixed'], apollo_probs, apollo_inputs, estimate_sett, outOfSample_settings)

    #ro.r('''
    #    out_of_sample_path = paste0(apollo_control$outputDirectory,'/',apollo_control$modelName, '_outOfSample_params.csv')
    #    validation = read.csv(out_of_sample_path)
    #    out_of_sample_LL = mean(validation$outOfSample_model)
    #    in_sample_LL = mean(validation$inSample_model)
    #    ''')

    # Now, we are estimating the model using the in_sample data to obtain the parameters and model fit.     
    ro.globalenv['database']            = ro.globalenv['in_sample']
    apollo_inputs                       = apollo.apollo_validateInputs()
    model                               = apollo.apollo_estimate(apollo_beta, ro.globalenv['apollo_fixed'], apollo_probs, apollo_inputs, estimate_sett)
    if info:
        apollo.apollo_modelOutput(model)
    ro.globalenv['model'] = model
    ro.globalenv['save_path'] = f"{save_directory}"
    apollo.apollo_saveOutput(model)
    
    ro.r('''
        model_summary <- data.frame(
                numParams = model$numParams,
                numResids = model$numResids,
                maximum = model$maximum,
                vcHessianConditionNumber = model$vcHessianConditionNumber,
                successfulEstimation = model$successfulEstimation,
                LL0 = model$LL0,
                LLC = model$LLC,
                LLout = model$LLout,
                rho2_0 = model$rho2_0,
                adjRho2_0 = model$adjRho2_0,
                rho2_C = model$rho2_C,
                adjRho2_C = model$adjRho2_C,
                AIC = model$AIC,
                BIC = model$BIC,
                eigValue = model$eigValue,
                timeTaken = model$timeTaken,
                nFreeParams = model$nFreeParams,
                #in_sample = in_sample_LL,
                #out_of_sample = out_of_sample_LL,
                t(data.frame(value = model$estimate, row.names = names(model$estimate))),
                t(data.frame(value = c(model$se, model$robse), row.names = c(paste0('se_', names(model$se)), paste0('rob_', names(model$robse)))))
        )
             
        cat('Saving model summary to ', save_path, '\n')

        write.csv(model_summary, file = save_path, row.names = FALSE)
    ''')
    return None

def create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates, info=False):

    num_alternatives = max(attributes.keys())
    num_attributes = max([attr for alt in attributes for attr in attributes[alt]])
        
    cov_keys = list(covariates.keys())
    position_to_cov = {i+1: cov_keys[i] for i in range(len(cov_keys))}

    for idx, cov_value in enumerate(covariates_0):
        if cov_value > len(cov_keys):
            raise ValueError(f"covariates_0 at position {idx} has value {cov_value}, but it should be between 1 and {len(cov_keys)} (or 0 for no covariate).")
            
    
    for idx, spec_value in enumerate(specific_0):
        if spec_value not in (0, 1):
            raise ValueError(f"specific_0 at position {idx} has value {spec_value}, but it should be 0 (generic) or 1 (specific).")
    
    allowed_state_values = {0, 1, 2, 3}
    for idx, state_value in enumerate(state_0):
        if state_value not in allowed_state_values:
            raise ValueError(f"state_0 at position {idx} has value {state_value}, but it should be one of {allowed_state_values}.")   
        
    if info:
            print(f'state_0: {state_0}')
            print(f'specific_0: {specific_0}')
            print(f'covariates_0: {covariates_0}')
            print(f'num_alternatives: {num_alternatives}')
            print(f'num_attributes: {num_attributes}')
            print(f'Covariate mapping (position -> name): {position_to_cov}\n')



    ####### ASC (Alternative-Specific Constants) #######
    # Build ASC options: basic and with covariate interactions.
    asc_basic       = [f'asc_alt{alt}' for alt in range(1, num_alternatives + 1)]
    asc_covariates  = [f'asc_alt{alt}_{cov}_{category}' for alt in range(1, num_alternatives + 1) for cov in covariates  for category in covariates[cov]]
    asc_options     = { 0: asc_basic + asc_covariates,                          # All fixed
                        1: [f'asc_alt{num_alternatives}'] + asc_covariates}     # ASC without interaction

    info and print(f'asc_option_index: {0} - {0} - {0}')
    info and print(f'asc_option_index: {1} - linear - {0}')

    # For interaction options, keys start at 2.
    for i, cov in enumerate(cov_keys, start=2):
            asc_covariates_excluding_i = [f'asc_alt{alt}_{c}_{category}' for alt in range(1, num_alternatives + 1) for c in covariates if c != cov for category in covariates[c]]
            asc_covariates_last = [f'asc_alt{num_alternatives}_{c}_{category}' for c in covariates if c == cov for category in covariates[c]]
            info and print(f'asc_option_index: {i} - linear - {cov}')
            asc_options[i] = asc_basic + asc_covariates_excluding_i + asc_covariates_last

    ####### Generic Options #######
    generic_basic = [f'b_{{}}_generic', f'b_{{}}_generic_log', f'b_{{}}_generic_box_cox']
    generic_covariates = [f'b_{{}}_generic{trans}_{cov}_{category}' for trans in ['', '_log', '_box_cox'] for cov in covariates for category in covariates[cov]]

    generic_options = { 0: generic_basic + [f'L_{{}}'] + generic_covariates,  # All fixed
                        1: [f'b_{{}}_generic_log', f'b_{{}}_generic_box_cox', f'L_{{}}'] + generic_covariates,  # Linear
                        2: [f'b_{{}}_generic', f'b_{{}}_generic_box_cox', f'L_{{}}'] + generic_covariates,  # Log
                        3: [f'b_{{}}_generic', f'b_{{}}_generic_log'] + generic_covariates  # Box-Cox
                        }
    info and print(f'\ngeneric_option_index: {0} - {0} - {0}')
    info and print(f'generic_option_index: {1} - linear - {0}')
    info and print(f'generic_option_index: {2} - log - {0}')
    info and print(f'generic_option_index: {3} - box_cox - {0}')
    option_index = 4
    for trans in ['', '_log', '_box_cox']:
        for cov in cov_keys:
            generic_covariates_excluding_trans = [term for term in generic_covariates  if not term.startswith(f'b_{{}}_generic{trans}_{cov}_')]
            info and print(f'generic_option_index: {option_index} - {trans if trans != "" else "linear"} - {cov}')
            if trans != '_box_cox':
                generic_covariates_excluding_trans += ['L_{}']
            generic_options[option_index] = generic_basic + generic_covariates_excluding_trans
            option_index += 1

    ####### Specific Options #######
    specific_options = {}
    for alt in range(1, num_alternatives + 1):
        specific_basic = [f'b_{alt}_{{}}', f'b_{alt}_{{}}_log', f'b_{alt}_{{}}_box_cox']
        specific_covariates = [f'b_{alt}_{{}}{trans}_{cov}_{category}' for trans in ['', '_log', '_box_cox'] for cov in covariates for category in covariates[cov]]
        specific_options[alt] = {   0: specific_basic + [f'L_{alt}_{{}}'] + specific_covariates,  # All fixed
                                    1: [f'b_{alt}_{{}}_log', f'b_{alt}_{{}}_box_cox', f'L_{alt}_{{}}'] + specific_covariates,  # Linear
                                    2: [f'b_{alt}_{{}}', f'b_{alt}_{{}}_box_cox', f'L_{alt}_{{}}'] + specific_covariates,  # Log
                                    3: [f'b_{alt}_{{}}', f'b_{alt}_{{}}_log'] + specific_covariates  # Box-Cox
                                    }
        info and print(f'\nspecific_option_index: {0} - {0} - {0}')
        info and print(f'specific_option_index: {1} - linear - {0}')
        info and print(f'specific_option_index: {2} - log - {0}')
        info and print(f'specific_option_index: {3} - box_cox - {0}')
        option_index = 4
        for trans in ['', '_log', '_box_cox']:
            for cov in cov_keys:
                specific_covariates_excluding_trans = [term for term in specific_covariates if not term.startswith(f'b_{alt}_{{}}{trans}_{cov}_')]
                if trans != '_box_cox':
                    specific_covariates_excluding_trans += [f'L_{alt}_{{}}']
                info and print(f'specific_option_index: {option_index} - {trans if trans != "" else "linear"} - {cov}')
                specific_options[alt][option_index] = specific_basic + specific_covariates_excluding_trans
                option_index += 1

    if info:
        print('\nasc_options:', list(asc_options.keys()))
        print('generic_options:', list(generic_options.keys()))
        for i in range(1, num_alternatives + 1):
            print(f'specific_options[{i}]:', list(specific_options[i].keys()))

    ####### Build Specification #######
    specification = []

    # ASC specification.
    if state_0[0] == 0:
        info and print(f'ASC is fixed -> [{state_0[0]}, {specific_0[0]}, {covariates_0[0]}]')
        specification.extend(asc_options[0])
    elif state_0[0] == 1:
        # Use the covariate setting for ASC: add 1 to map to keys starting at 2.
        key_for_asc = covariates_0[0] + 1  
        specification.extend(asc_options.get(key_for_asc, []))

    # For each attribute (attributes 1...num_attributes).
    for attr in range(1, num_attributes + 1):
            if state_0[attr] == 0:  # Fixed
                info and print(f'attr: {attr} is fixed -> [{state_0[attr]}, {specific_0[attr]}, {covariates_0[attr]}]')
                specification.extend([param.format(attr) for param in generic_options[0]])
                specification.extend([param.format(attr) for key in range(1, num_alternatives + 1) for param in specific_options[key][0]])
            else:
                if specific_0[attr] == 0:  # Generic + no covariate
                    if covariates_0[attr] == 0:
                        info and print(f'attr: {attr} is generic and has no covariate -> [{state_0[attr]}, {specific_0[attr]}, {covariates_0[attr]}]')
                        specification.extend([param.format(attr) for param in generic_options[state_0[attr]]])
                        specification.extend([param.format(attr) for key in range(1, num_alternatives + 1)  for param in specific_options[key][0]])                        

                    else: # Generic + Covariate
                        info and print(f'attr: {attr} is generic and has covariate -> [{state_0[attr]}, {specific_0[attr]}, {covariates_0[attr]}]')
                        if state_0[attr] == 1:
                            base = 4    # linear: indices 4..10
                        elif state_0[attr] == 2:
                            base = 11   # log: indices 11..17
                        elif state_0[attr] == 3:
                            base = 18   # box-cox: indices 18..24
                        index = base + (covariates_0[attr] - 1)
                        info and print(f'generic + covariate index: {index} - interacting wih {position_to_cov[covariates_0[attr]]}\n')
                        specification.extend([param.format(attr) for param in generic_options[index]])
                        specification.extend([param.format(attr) for key in range(1, num_alternatives + 1) for param in specific_options[key][0]])

                else:  # Specific
                    for alt in range(1, num_alternatives + 1):
                            if alt in attributes and attr not in attributes[alt]:
                                info and print(f'attr: {attr} is specific and not in alt: {alt}')
                                specification.extend([param.format(attr) for param in specific_options[alt][0]])
                                if alt == num_alternatives:
                                    specification.extend([param.format(attr) for param in generic_options[0]])
                            else:
                                if covariates_0[attr] == 0: # Specific + no covariate
                                    info and print(f'attr: {attr} is specific and has no covariate -> [{state_0[attr]}, {specific_0[attr]}, {covariates_0[attr]}]')
                                    specification.extend([param.format(attr) for param in specific_options[alt][state_0[attr]]])
                                    if alt == num_alternatives:
                                        specification.extend([param.format(attr) for param in generic_options[0]])
                                else: # Specific + Covariate
                                    info and print(f'attr: {attr} is specific and has covariate -> [{state_0[attr]}, {specific_0[attr]}, {covariates_0[attr]}]\n')
                                    if state_0[attr] == 1:
                                        base = 4
                                    elif state_0[attr] == 2:
                                        base = 11
                                    elif state_0[attr] == 3:
                                        base = 18
                                    index = base + (covariates_0[attr] - 1)
                                    info and print(f'specific index: {index} - interacting wih {position_to_cov[covariates_0[attr]]}')
                                    specification.extend([param.format(attr) for param in specific_options[alt][index]])
                                    if alt == num_alternatives:
                                        specification.extend([param.format(attr) for param in generic_options[0]])

    return specification 

def create_apollo_non_fixed(beta_fixed, attributes, covariates, info=False):  
    
    num_attributes = max(max(attr_list) for attr_list in attributes.values())

    state_0 = [0] *  (num_attributes + 1)  # Initial state (assume only ASC enabled)
    specific_0 = [0] * (num_attributes + 1)  # Assume generic by default
    covariates_0 = [0] * (num_attributes + 1)  # No covariates initially

    all_parameters = create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates, info)

    estimated_parameters = [param for param in all_parameters if param not in beta_fixed]

    return estimated_parameters

def clean_apollo_outputs(job_output_dir, name):
    """
    Delete intermediate Apollo-generated files for the given model specification.
    """
    patterns = [
        f"{name}_estimates.csv",
        f"{name}_output.txt",
        f"{name}_results.csv",
        f"{name}_iterations.csv",
        f"{name}_model.rds",
        f"{name}_outOfSample_params.csv",
        f"{name}_outOfSample_samples.csv"
    ]
    for pattern in patterns:
        files_to_delete = glob.glob(os.path.join(job_output_dir, pattern))
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                os.remove(file_path)

def get_mnl_outcomes(state_0, specific_0, covariates_0, attributes, covariates, job_index, path_dir, path_df, r=True, info=True):
        
        # Define job-specific output
        job_output_dir = os.path.join(path_dir, "outputs", f"job_{job_index}")
        os.makedirs(job_output_dir, exist_ok=True)
        
        # Unique rewards file for this job index
        folder_path = os.path.join(path_dir, "outputs")
        job_results_file = os.path.join(folder_path, f"rewards_{job_index}.csv")

        # Load agent-specific rewards file
        if os.path.exists(job_results_file):
            job_df = pd.read_csv(job_results_file)
        else:
            job_df = pd.DataFrame(columns=["specification"])

        # Build specification string
        specification = [f"{state}{specific}{cov}" for state, specific, cov in zip(state_0, specific_0, covariates_0)]
        name = "_".join(specification)

        # Check if model already estimated
        if name in job_df['specification'].values:
            row = job_df[job_df['specification'] == name]
            return row
        else:
            try:
                apollo_beta_fixed = create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates)
                apollo_beta_estimated = create_apollo_non_fixed(apollo_beta_fixed, attributes, covariates)
                mnl_interaction(name, apollo_beta_fixed, apollo_beta_estimated, attributes, covariates, job_index, path_dir, path_df, info)

                new_outcomes_path = os.path.join(job_output_dir, f"{name}_results.csv")
                new_outcomes = pd.read_csv(new_outcomes_path)
                new_outcomes['specification'] = name
                new_outcomes = new_outcomes[['specification'] + [col for col in new_outcomes.columns if col != 'specification']]

            except Exception as e:
                info and print(f"Error in model {name}: {e}")
                new_outcomes = pd.DataFrame([{**{col: np.nan for col in job_df.columns if col != 'specification'}, 'specification': name}])
                new_outcomes['successfulEstimation'] = False

            # Update agent-specific rewards file
            updated_job_df = pd.concat([job_df, new_outcomes], ignore_index=True).drop_duplicates(subset=['specification'])
            updated_job_df.to_csv(job_results_file, index=False)

            clean_apollo_outputs(job_output_dir, name)

            return new_outcomes

def get_mnl_outcomes_parallel(state_0, specific_0, covariates_0, attributes, covariates, job_index, path_dir, path_df, batch_df, r=True, info=True):
    
    # Define job-specific output
    job_output_dir = os.path.join(path_dir, "outputs", f"job_{job_index}")
    os.makedirs(job_output_dir, exist_ok=True)
        
    # Unique rewards file for this job index
    folder_path = os.path.join(path_dir, "outputs")
    job_results_file = os.path.join(folder_path, f"rewards_{job_index}.csv")

    # Load agent-specific rewards file
    if os.path.exists(job_results_file):
        job_df = pd.read_csv(job_results_file)
    else:
        job_df = pd.DataFrame(columns=["specification"])

    # Build specification string
    specification = [f"{state}{specific}{cov}" for state, specific, cov in zip(state_0, specific_0, covariates_0)]
    name = "_".join(specification)


    # Check if model already estimated (in batch or file)
    if name in batch_df['specification'].values:
        return batch_df[batch_df['specification'] == name]
    
    folder_path = os.path.join(path_dir, "outputs")
    job_results_file = os.path.join(folder_path, f"rewards_{job_index}.csv")
    
    if os.path.exists(job_results_file):
        job_df = pd.read_csv(job_results_file)
    else:
        job_df = pd.DataFrame(columns=["specification"])

    if name in job_df['specification'].values:
        return job_df[job_df['specification'] == name]

    # Estimate model
    try:
        apollo_beta_fixed = create_apollo_fixed(state_0, specific_0, covariates_0, attributes, covariates)
        apollo_beta_estimated = create_apollo_non_fixed(apollo_beta_fixed, attributes, covariates)
        mnl_interaction(name, apollo_beta_fixed, apollo_beta_estimated, attributes, covariates, job_index, path_dir, path_df, info)

        job_output_dir = os.path.join(path_dir, "outputs", f"job_{job_index}")
        new_outcomes_path = os.path.join(job_output_dir, f"{name}_results.csv")
        new_outcomes = pd.read_csv(new_outcomes_path)
        new_outcomes['specification'] = name
        new_outcomes = new_outcomes[['specification'] + [col for col in new_outcomes.columns if col != 'specification']]
    
    except Exception as e:
        info and print(f"Error in model {name}: {e}")
        new_outcomes = pd.DataFrame([{**{col: np.nan for col in job_df.columns if col != 'specification'}, 'specification': name}])
        new_outcomes['successfulEstimation'] = False

    # Add to batch_df
    batch_df = pd.concat([batch_df, new_outcomes], ignore_index=True).drop_duplicates(subset=['specification'])

    clean_apollo_outputs(job_output_dir, name)

    return batch_df
    
def generate_models_V1(num_attributes):
    asc = [0, 1]
    att = [0, 1, 2, 3]
    spec = [0]
    cov = [0]

    combinations_list = [
            list(combination) for combination in itertools.product(
                asc,
                *(att for _ in range(num_attributes)),  # Attributes
                *(spec for _ in range(num_attributes)), # Specific states
                *(cov for _ in range(num_attributes))  # Covariate states
            )
        ] 

    combinations_list = [comb for comb in combinations_list if sum(comb[:1 + num_attributes]) > 0]

    combinations_list = [comb[:1 + num_attributes] + [0] + comb[1 + num_attributes:] + [0] for comb in combinations_list]

    columns = (['asc'] + [f'att{i + 1}' for i in range(num_attributes)] + [f'spec{i}' for i in range(num_attributes+1)] + [f'cov{i}' for i in range(num_attributes+1)])

    combinations_df = pd.DataFrame(combinations_list, columns=columns)

    combinations_df['state']     = combinations_df.apply(lambda row: [row['asc']] + [row[f'att{i + 1}'] for i in range(num_attributes)], axis=1 )
    combinations_df['specific']  = combinations_df.apply(lambda row: [row[f'spec{i}'] for i in range(num_attributes+1)], axis=1 )
    combinations_df['covariate'] = combinations_df.apply(lambda row: [row[f'cov{i}'] for i in range(num_attributes+1)], axis=1 )

    return combinations_df

def generate_models_V2(num_attributes):
    asc = [0, 1]
    att = [0, 1, 2, 3]
    spec = [0, 1]
    cov = [0]

    combinations_list = [
        list(combination) for combination in itertools.product(
            asc,
            *(att for _ in range(num_attributes)),  # Attributes
            *(spec for _ in range(num_attributes)),  # Specific states
            *(cov for _ in range(num_attributes))  # Covariate states
        )
    ]    
    combinations_list = [comb for comb in combinations_list if sum(comb[:1 + num_attributes]) > 0]
    combinations_list = [comb[:1 + num_attributes] + [0] + comb[1 + num_attributes:] + [0] for comb in combinations_list]

    columns = (['asc'] + [f'att{i + 1}' for i in range(num_attributes)] + [f'spec{i}' for i in range(num_attributes+1)] + [f'cov{i}' for i in range(num_attributes+1)])

    combinations_df = pd.DataFrame(combinations_list, columns=columns)
    combinations_df['state_0'] = combinations_df.apply(lambda row: [row['asc']] + [row[f'att{i + 1}'] for i in range(num_attributes)], axis=1 )
    combinations_df['state_0'] = combinations_df['state'].apply(lambda x: [int(i) for i in x])
    combinations_df['specific_0'] = combinations_df.apply(lambda row: [row[f'spec{i}'] for i in range(num_attributes+1)], axis=1 )
    combinations_df['covariates_0'] = combinations_df.apply(lambda row: [row[f'cov{i}'] for i in range(num_attributes+1)], axis=1 )

    return combinations_df

def generate_models_V3(num_attributes, covariates, max_cov):   
    asc  = [0, 1]
    att  = [0, 1, 2, 3]
    spec = [0, 1]
    cov  = list(range(0, len(covariates.keys()) + 1))

    # Generate all combinations
    combinations_list = [
        list(combination) for combination in itertools.product(
            asc,
            *(att for _ in range(num_attributes)),  # Attributes

            [0], 
            *(spec for _ in range(num_attributes)),  # Specific states

            *(cov for _ in range(num_attributes+1))  # Covariate states
        )
    ]    

    valid_combinations = [comb for comb in combinations_list if sum(comb[:1 + num_attributes]) > 0]

    # Apply constraint: Covariates are used at most max_cov times
    constrained_combinations = []
    for comb in valid_combinations:
        covariate_counts = {cov_idx: 0 for cov_idx in range(1, len(covariates) + 1)}
        for cov_value in comb[-(num_attributes + 1):]:  
            if cov_value > 0:
                covariate_counts[cov_value] += 1

        if all(count <= max_cov for count in covariate_counts.values()):
            constrained_combinations.append(comb)

    columns = (['asc'] +  [f'att{i + 1}' for i in range(num_attributes)] +  [f'spec{i}' for i in range(num_attributes+1)] + [f'cov{i}' for i in range(num_attributes+1)])
    combinations_df = pd.DataFrame(constrained_combinations, columns=columns)
    combinations_df['state'] = combinations_df.apply(lambda row: [row['asc']] + [row[f'att{i + 1}'] for i in range(num_attributes)], axis=1)
    combinations_df['specific'] = combinations_df.apply(lambda row: [row[f'spec{i}'] for i in range(num_attributes+1)], axis=1)
    combinations_df['covariate'] = combinations_df.apply(lambda row: [row[f'cov{i}'] for i in range(num_attributes+1)], axis=1)

    return combinations_df

def generate_covariate_subsets(covariates, r):
    covariate_keys = list(covariates.keys())
    subsets = list(itertools.combinations(covariate_keys, r))
    subset_dicts = [{key: covariates[key] for key in subset} for subset in subsets]
    return subset_dicts

   ### Time taken for each model specification and identification of best candidates
def timeTaken(training_log_path, rewards_path):    
    if not os.path.exists(training_log_path):
        raise FileNotFoundError(f"Training log not found at: {training_log_path}")
    
    if not os.path.exists(rewards_path):
        raise FileNotFoundError(f"Rewards file not found at: {rewards_path}")

    modelling_outcomes = pd.read_csv(rewards_path)
    training_df = pd.read_csv(training_log_path)
    specifications = training_df['specification'].tolist()

    # Add timeTaken
    time_taken_list = []
    for spec in specifications:
        row = modelling_outcomes[modelling_outcomes['specification'] == spec]
        if not row.empty:
            time_taken_list.append(row.iloc[0].get('timeTaken', None))
        else:
            time_taken_list.append(None)
    training_df['timeTaken'] = time_taken_list

    # Identify and index best candidates
    max_reward_so_far = float('-inf')
    candidate_index = 0
    best_candidate_indices = []

    for reward in training_df['reward']:
        if reward > max_reward_so_far:
            candidate_index += 1
            best_candidate_indices.append(candidate_index)
            max_reward_so_far = reward
        else:
            best_candidate_indices.append(0)

    training_df['bestCandidateIndex'] = best_candidate_indices
    training_df.to_csv(training_log_path, index=False)

# Function to calculate exploration metrics
def exploration(episode, training_log_path):
    if not os.path.exists(training_log_path):
        raise FileNotFoundError(f"Training log not found at: {training_log_path}")

    df = pd.read_csv(training_log_path)

    if 'episode' not in df.columns or 'timeTaken' not in df.columns or 'specification' not in df.columns:
        raise ValueError("The training log must contain 'episode', 'timeTaken', and 'specification' columns.")

    # Filter up to the specified episode
    df = df[df['episode'] <= episode].copy()

    # Clean non-numeric values in timeTaken
    df['timeTaken'] = pd.to_numeric(df['timeTaken'], errors='coerce')
    df = df.dropna(subset=['timeTaken'])

    num_episodes = len(df)
    unique_specs = df['specification'].nunique()
    unique_model_ratio = unique_specs / num_episodes if num_episodes > 0 else 0

    # get max  index reward considering the bestCandidateIndex column
    max_index = df['bestCandidateIndex'].max()
    max_index_row = df[df['bestCandidateIndex'] == max_index].iloc[0]
    max_index_spec = max_index_row['specification']
    

    result = {
        'mean_time': round(df['timeTaken'].mean(), 3),
        'std_time': round(df['timeTaken'].std(), 3),
        'sum_time': round(df['timeTaken'].sum()/60, 3),
        'unique_models': round(unique_model_ratio, 4),
        'episode': episode,
        'best_candidate': max_index_spec,
        'max_reward': df['reward'].max()        
    }

    return result



