"""
    Script to check google colab memory usage
"""

import os

import GPUtil as GPU
import humanize
import psutil

GPUs = GPU.getGPUs()

# XXX: only one GPU on Colab and isn’t guaranteed
gpu = GPUs[0]


def available_gpu():
    process = psutil.Process(os.getpid())
    print("*" * 66)
    print(
        "\nGen RAM Free: "
        + humanize.naturalsize(psutil.virtual_memory().available),
        " | Proc size: " + humanize.naturalsize(process.memory_info().rss),
    )
    print(
        "GPU RAM Free: {0:.2f}GB | Used: {1:.2f}GB | Util {2:3.2f}% | Total {3:.2f}GB\n".format(
            gpu.memoryFree / 1024,
            gpu.memoryUsed / 1024,
            gpu.memoryUtil * 100,
            gpu.memoryTotal / 1024,
        )
    )
    print("*" * 66)


available_gpu()
