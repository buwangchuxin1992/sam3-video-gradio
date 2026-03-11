# sam3-video-gradio
Based on the SAM3 video tracking and segmentation system, it can be used for point-prompt-based video segmentation, suitable for both digital image (visible light) annotation and especially applicable to infrared video segmentation. Leveraging SAM3’s powerful segmentation generalization capability, this system achieves excellent results in infrared video segmentation. However, as infrared videos lack color information, SAM3’s text-prompt performance is suboptimal. Hence, this system specifically employs point-prompt annotation to fully exploit SAM3’s capabilities.

基于sam3的视频跟踪分割系统，可用于基于点提示的视频分割，既可以用于数码图像（可见光）标注，也特别适用于红外视频分割。借助于sam3强大的分割泛化能力，本系统在红外视频分割中有很好的效果。同时，红外无颜色信息，使用sam3的文本提示效果不佳，故这里特开发基于点提示标注，从而充分发挥sam3的能力。


Acknowledgments

The code are largerly borred from  https://github.com/Pytorchlover/sam3-gradio We are grateful for the helpful resources provided
