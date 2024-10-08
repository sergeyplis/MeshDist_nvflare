from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
import os

def get_data_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the data directory path based on the available paths."""
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    
    # # Production path (check environment variable or default to /workspace)
    # production_path = os.getenv("DATA_DIR", "/workspace/data")
    # if os.path.exists(production_path):
    #     return production_path
    
    # Simulator path
    simulator_path = os.path.abspath(os.path.join(os.getcwd(), "../../../test_data", site_name))

    if os.path.exists(simulator_path):
        return simulator_path
    
    # # POC path
    # poc_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../test_data", site_name))
    # if os.path.exists(poc_path):
    #     return poc_path
    
    # Raise an error if no path is found and include attempted paths
    raise FileNotFoundError(f"Data directory path could not be determined. "
                            f"Attempted paths:\nSimulator path: {simulator_path}")

def get_output_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the output directory path based on the available paths."""
    job_id = fl_ctx.get_job_id()
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    
    # # Production path (check environment variable or default to /workspace)
    # production_path = os.getenv("OUTPUT_DIR", "/workspace/output")
    # if os.path.exists(production_path):
    #     return production_path
    
    # Simulator path
    simulator_path = os.path.abspath(os.path.join(os.getcwd(), "../../../test_output", job_id, site_name))
    if not os.path.exists(simulator_path):
        os.makedirs(simulator_path, exist_ok=True)
        
    return simulator_path
    
    # # POC path
    # poc_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../test_output", job_id, site_name))
    # if os.path.exists(poc_path):
    #     os.makedirs(poc_path, exist_ok=True)
    #     return poc_path
    
    # Raise an error if no path is found
    raise FileNotFoundError(f"output directory path could not be determined."
                            f"Attempted paths:\nSimulator path: {simulator_path}")

def get_parameters_file_path(fl_ctx: FLContext) -> str:
    """Determine and return the parameters file path based on the available paths."""
    
    # Production path (check environment variable or default to /workspace)
    production_path = os.getenv("PARAMETERS_FILE_PATH", "/workspace/runKit/parameters.json")
    if os.path.exists(production_path):
        return production_path
    
    # Simulator path
    simulator_path = os.path.abspath(os.path.join(os.getcwd(), "../test_data", "server", "parameters.json"))
    if os.path.exists(simulator_path):
        return simulator_path
    
    # POC path
    poc_path = os.path.abspath(os.path.join(os.getcwd(), "../../../../test_data", "server", "parameters.json"))
    if os.path.exists(poc_path):
        return poc_path
    
    # Raise an error if no path is found
    raise FileNotFoundError("Parameters file path could not be determined.")