## Overview
This repository contains the codebase for a full-stack system that uses Wi-Fi Channel State Information (CSI) to classify and monitor human activities completely without cameras or wearables. The system leverages a Vision Transformer (ViT) for accurate activity recognition and features a complete user interface for real-time monitoring and notifications. 

Additionally, this project includes comprehensive scripts for synthetic data generation used to train, test, and expand the UT-HAR dataset.

This project is built upon the **SenseFi (WiFi-CSI-Sensing-Benchmark)** framework, expanding its core capabilities with advanced generative models and a complete GUI for practical HAR experiments.

## What's New in This Repository?
While this project utilizes the core data processing and modeling pipelines from SenseFi, the following minor additions and modifications have been made:
* **Synthetic Data Generation:** Implementation of a diffusion model to generate synthetic Wi-Fi CSI data, successfully expanding the UT-HAR dataset.
* **Front-End Integration:** An interactive HTML/JS dashboard connected via a Python backend to visualize AI predictions in real-time.
* **Mobile Alerts:** Automated notification features pushed directly to Android devices based on live activity triggers.

## Acknowledgements & Attribution
The foundational Wi-Fi sensing benchmark and data extraction tools in this repository are modified from the original [SenseFi repository](https://github.com/xyanchen/wifi-csi-sensing-benchmark). 

> Jianfei Yang et al., *SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing*, PRCV 2023.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
