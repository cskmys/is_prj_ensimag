import pynvml as nv
import global_const as gc


def get_gpu_info():
    nv.nvmlInit()
    gc.prj.misc.gpu_info.driver_ver = nv.nvmlSystemGetDriverVersion().decode()
    gpu_cnt = nv.nvmlDeviceGetCount()
    gc.prj.misc.gpu_info.nb_gpu = gpu_cnt
    for i in range(gpu_cnt):
        handle = nv.nvmlDeviceGetHandleByIndex(i)
        gc.prj.misc.gpu_info.gpu_name = nv.nvmlDeviceGetName(handle).decode()
        mem_info = nv.nvmlDeviceGetMemoryInfo(handle)
        gc.prj.misc.gpu_info.gpu_mem_mb = mem_info.total / 1024 / 1024
        gc.prj.misc.gpu_info.clk_info = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_GRAPHICS)
    nv.nvmlShutdown()
