from scripts.portable_inference_runtime import install
install()
import sys
if sys.argv[0] == "-m" or sys.argv[0].endswith("/scripts/finish_fusion_100h.py"):
    from scripts.future_speed_runtime import install as speed
    speed()
