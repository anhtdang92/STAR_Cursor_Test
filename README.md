# Video Upscaler

A full-stack application for AI-powered video upscaling, built with React and Flask. This project implements the STAR (Spatial-Temporal Augmentation) model for high-quality video enhancement.

## 🌟 Features

- **Video Processing**
  - Upload and process MP4 videos (up to 500MB)
  - Multiple upscaling options (2x, 4x, 8x)
  - AI model selection (Artemis, Gaia, Theia)
  - Denoise and detail enhancement controls
  - Real-time processing status
  - Progress tracking and ETA

- **User Interface**
  - Modern Apple-style design
  - Responsive layout
  - Real-time video preview
  - Intuitive processing controls
  - Progress visualization

## 🛠️ Tech Stack

### Frontend
- React with TypeScript
- Styled Components
- Tailwind CSS
- Axios for API calls

### Backend
- Python Flask
- FFmpeg for video processing
- STAR AI model for upscaling
- PyTorch for deep learning

## 📋 Prerequisites

- Node.js 16+
- Python 3.8+
- FFmpeg
- Git
- GPU with 24GB+ VRAM (recommended)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd video-upscaler
```

2. Set up the frontend:
```bash
cd frontend
npm install
npm start
```

3. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

4. Create required directories:
```bash
mkdir backend/uploads backend/processed
```

## 💻 Usage

1. Open your browser and navigate to `http://localhost:3000`
2. Upload an MP4 video file (max 500MB)
3. Configure processing settings:
   - Select upscaling factor (2x, 4x, 8x)
   - Choose AI model (Artemis, Gaia, Theia)
   - Adjust denoise level
   - Toggle detail enhancement
4. Click "Process" to start upscaling
5. Monitor progress in real-time
6. Download the processed video when complete

## 🔧 Development

### Running Tests
```bash
# Frontend tests
cd frontend
npm test

# Backend tests
cd backend
python -m pytest
```

### PowerShell Command Syntax
When running commands in PowerShell, use semicolons (;) instead of && for command chaining:
```powershell
# Correct PowerShell syntax
cd backend; python app.py

# Incorrect syntax (will cause error)
cd backend && python app.py
```

### Activating the Virtual Environment
Before running any commands, ensure that the virtual environment is activated:
```powershell
# Activate the virtual environment
.venv\\Scripts\\activate
```

### Building for Production
```bash
cd frontend
npm run build
```

## 📊 Performance

- Processing time depends on:
  - Video length
  - Resolution
  - GPU capabilities
  - Selected upscaling factor
- Recommended specifications:
  - GPU: NVIDIA with 24GB+ VRAM
  - RAM: 32GB+
  - Storage: SSD recommended

## 🔒 Security

- File size limits enforced
- Secure file handling
- Input validation
- Error handling and logging
- Automatic cleanup of temporary files

## 📝 Notes

- Maximum video file size: 500MB
- Only MP4 files are supported
- Processing time varies based on video length and GPU capabilities
- Temporary files are automatically cleaned up after processing
- GPU memory requirements increase with higher upscaling factors

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For any inquiries, please reach out via email at `ruixie0097@gmail.com`

## 🙏 Acknowledgments

This project is based on:
- [I2VGen-XL](https://github.com/ali-vilab/VGen)
- [VEnhancer](https://github.com/Vchitect/VEnhancer)
- [CogVideoX](https://github.com/THUDM/CogVideo)
- [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M)

## 📚 Citation

If this project helps your research or work, please consider citing our paper:

```bibtex
@misc{xie2025starspatialtemporalaugmentationtexttovideo,
      title={STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution}, 
      author={Rui Xie and Yinhong Liu and Penghao Zhou and Chen Zhao and Jun Zhou and Kai Zhang and Zhenyu Zhang and Jian Yang and Zhenheng Yang and Ying Tai},
      year={2025},
      eprint={2501.02976},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.02976}, 
}
```

<div align="center">
    <h1>
    STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution
    </h1>
    <div>
        <a href='https://github.com/CSRuiXie' target='_blank'>Rui Xie<sup>1*</sup></a>,&emsp;
        <a href='https://github.com/yhliu04' target='_blank'>Yinhong Liu<sup>1*</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=yWq1Fd4AAAAJ' target='_blank'>Penghao Zhou<sup>2</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=Uhp3JKgAAAAJ&hl=zh-CN&oi=sra' target='_blank'>Chen Zhao<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=w03CHFwAAAAJ' target='_blank'>Jun Zhou<sup>3</sup></a><br>
        <a href='https://cszn.github.io/' target='_blank'>Kai Zhang<sup>1</sup></a>,&emsp;
        <a href='https://jessezhang92.github.io/' target='_blank'>Zhenyu Zhang<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com.hk/citations?user=6CIDtZQAAAAJ&hl=zh-CN' target='_blank'>Jian Yang<sup>1</sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?hl=zh-CN&user=Ds5wwRoAAAAJ' target='_blank'>Zhenheng Yang<sup>2</sup></a>,&emsp;
        <a href='https://tyshiwo.github.io/index.html' target='_blank'>Ying Tai<sup>1&#8224</sup></a>
    </div>
    <div>
        <sup>1</sup>Nanjing University,&emsp;<sup>2</sup>ByteDance,&emsp; <sup>3</sup>Southwest University
    </div>
    <div>
        <h4 align="center">
            <a href="https://nju-pcalab.github.io/projects/STAR" target='_blank'>
                <img src="https://img.shields.io/badge/🌟-Project%20Page-blue" style="padding-right: 20px;">
            </a>
            <a href="https://arxiv.org/abs/2501.02976" target='_blank'>
                <img src="https://img.shields.io/badge/arXiv-2501.02976-b31b1b.svg" style="padding-right: 20px;">
            </a>
            <a href="https://youtu.be/hx0zrql-SrU" target='_blank'>
                <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white" style="padding-right: 20px;">
            </a>
            <br>
            <a href="https://huggingface.co/spaces/SherryX/STAR" target='_blank'>
                <img src="https://img.shields.io/static/v1?label=Demo STAR&message=HuggingFace&color=yellow">
            </a>
            <a href="https://colab.research.google.com/drive/1K8A1U_BNpAteRhhW9A8pAYs6LWjItQs_?usp=sharing" target='_blank'>
                <img src="https://img.shields.io/static/v1?label=Demo STAR&message=Google Colab&color=orange">
            </a>
        </h4>
    </div>
</div>


## 🔆 Updates
- **2025.01.19** The STAR demo is now available on [Google Colab](https://colab.research.google.com/drive/1K8A1U_BNpAteRhhW9A8pAYs6LWjItQs_?usp=sharing). Feel free to give it a try!

- **2025.01.09** The online demo of STAR on [Hugging Face](https://huggingface.co/spaces/SherryX/STAR) is now live! Please note that due to the duration limitation of ZeroGPU, the running time may exceed the allocated GPU duration. If you'd like to try it, you can duplicate the demo and assign a paid GPU.

- **2025.01.07**  The pretrained STAR model (I2VGen-XL and CogVideoX-5B versions) and inference code have been released.


## 📑 TODO
- [ ] Training codes
- [x] Inference codes
- [x] Online demo


## 🔎 Method Overview
![STAR](assets/overview.png)


## 📷 Results Display
![STAR](assets/teaser.png)
![STAR](assets/real_world.png)
👀 More visual results can be found in our [Project Page](https://nju-pcalab.github.io/projects/STAR) and [Video Demo](https://youtu.be/hx0zrql-SrU).


## ⚙️ Dependencies and Installation
**VRAM requirement**: Upscaling the provided toy example by 4x, with 72 frames, a width of 426, and a height of 240, requires around 39GB of VRAM using the default settings. If you encounter an OOM problem, you can set a smaller frame_length in inference_sr.sh. We recommend using a GPU with at least 24GB of VRAM to run this project. 

```README.md
.
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── components/
│   └── package.json
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── temp/
│       ├── uploads/
│       └── processed/
└── README.md
```

## Notes

- Maximum video file size: 500MB
- Only MP4 files are supported
- Processing time depends on video length and GPU capabilities
- Temporary files are automatically cleaned up after processing

## License

This project uses the STAR model which is subject to its own license terms. Please refer to the original repository for license information.

## 🚀 Inference

### Model Weight
| Base Model | Type | URL |
|------------|--------|-----------------------------------------------------------------------------------------------|
| I2VGen-XL | Light Degradation | [:link:](https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/light_deg.pt?download=true) |
| I2VGen-XL | Heavy Degradation | [:link:](https://huggingface.co/SherryX/STAR/resolve/main/I2VGen-XL-based/heavy_deg.pt?download=true) |
| CogVideoX-5B | Heavy Degradation | [:link:](https://huggingface.co/SherryX/STAR/tree/main/CogVideoX-5B-based) |

### 1. I2VGen-XL-based 
#### Step 1: Download the pretrained model STAR from [HuggingFace](https://huggingface.co/SherryX/STAR).
We provide two versions for I2VGen-XL-based model, `heavy_deg.pt` for heavy degraded videos and `light_deg.pt` for light degraded videos (e.g., the low-resolution video downloaded from video websites).

You can put the weight into `pretrained_weight/`.

#### Step 2: Prepare testing data
You can put the testing videos in the `input/video/`.

As for the prompt, there are three options: 1. No prompt. 2. Automatically generate a prompt (e.g., [using Pllava](https://github.com/hpcaitech/Open-Sora/tree/main/tools/caption#pllava-captioning)). 3. Manually write the prompt. You can put the txt file in the `input/text/`.


#### Step 3: Change the path
You need to change the paths in `video_super_resolution/scripts/inference_sr.sh` to your local corresponding paths, including `video_folder_path`, `txt_file_path`, `model_path`, and `save_dir`.


#### Step 4: Running inference command
```
bash video_super_resolution/scripts/inference_sr.sh
```

### 2. CogVideoX-based
Refer to these [instructions](https://github.com/NJU-PCALab/STAR/tree/main/cogvideox-based#cogvideox-based-model-inference) for inference with the CogVideX-5B-based model.

Please note that the CogVideX-5B-based model supports only 720x480 input.

## ❤️ Acknowledgments
This project is based on [I2VGen-XL](https://github.com/ali-vilab/VGen), [VEnhancer](https://github.com/Vchitect/VEnhancer), [CogVideoX](https://github.com/THUDM/CogVideo) and [OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M). Thanks for their awesome works.


## 🎓Citations
If our project helps your research or work, please consider citing our paper:

```
@misc{xie2025starspatialtemporalaugmentationtexttovideo,
      title={STAR: Spatial-Temporal Augmentation with Text-to-Video Models for Real-World Video Super-Resolution}, 
      author={Rui Xie and Yinhong Liu and Penghao Zhou and Chen Zhao and Jun Zhou and Kai Zhang and Zhenyu Zhang and Jian Yang and Zhenheng Yang and Ying Tai},
      year={2025},
      eprint={2501.02976},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.02976}, 
}
```


## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `ruixie0097@gmail.com`


## 📄 License
I2VGen-XL-based models are distributed under the terms of the [MIT License](https://choosealicense.com/licenses/mit/).

CogVideoX-5B-based model is distributed under the terms of the [CogVideoX License](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/LICENSE).
