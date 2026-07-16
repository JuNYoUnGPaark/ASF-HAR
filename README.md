# Dynamics-Aware Action State Flow for Wearable Human Activity Recognition

<p align="center"><img src='./overall.png'></p>

This repository implements the methodology proposed in the paper "Dynamics-Aware Action State Flow for Wearable Human Activity Recognition".


## Paper Overview
**Abstract**: Deploying human activity recognition (HAR) on wearable IoT devices requires representations that remain reliable under tight compute/energy budgets and gradual sensor degradation over long-term use. In this setting, slowly varying low-frequency bias can distort absolute signal levels and degrade static, snapshot-based features. We propose Action State Flow (ASF), a dynamics-centric formulation that represents an activity via its latent state evolution rather than instantaneous latent states. ASF defines a flow representation as the first-order latent transition, which mitigates baseline shifts by emphasizing relative evolution, and factorizes the flow into direction and magnitude to disentangle intrinsic dynamical patterns from nuisance scaling. State differencing suppresses additive low-frequency components, while direction normalization reduces sensitivity to multiplicative scale changes, yielding a transition-centric flow field tailored to drift-prone sensing. To further structure these dynamics for classification, ASF introduces a flow–prototype regularization objective with class-conditioned prototypes, encouraging consistent transition geometry within each activity while maintaining inter-class separability. Across five public benchmarks (UCI-HAR, WISDM, UniMiB, PAMAP2, and MHEALTH), ASF matches strong baselines in standard accuracy and improves robustness under low-frequency distortions compared with static representations. We also assess deployability on a Raspberry Pi 3B+, demonstrating practical end-to-end on-device inference latency and memory footprint for real-time wearable HAR. Finally, INT8 quantization reduces model size by approximately 71\% with only a marginal accuracy drop and improved runtime efficiency, supporting ASF as a lightweight approach for robust HAR under signal variations on edge devices.

## Dataset
- **UCI-HAR** dataset is available at _https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones_
- **PAMAP2** dataset is available at _https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring_
- **MHEALTH** dataset is available at _https://archive.ics.uci.edu/dataset/319/mhealth+dataset_
- **WISDM** dataset is available at _https://www.cis.fordham.edu/wisdm/dataset.php_
- **UniMiB** dataset is available at _http://www.sal.disco.unimib.it/technologies/unimib-shar/_

## Requirements
```
torch==2.5.0+cu126
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
matplotlib==3.10.0
seaborn==0.13.2
fvcore==0.1.5.post20221221
```
To install all required packages:
```
pip install -r requirements.txt
```

## Codebase Overview
- `model.py` - Implementation of the proposed **ASF** architecture.
The implementation uses PyTorch, Numpy, pandas, scikit-learn, matplotlib, seaborn, and fvcore (for FLOPs analysis).

## Citation

If you use this work, please cite:

```bibtex
@article{park2026dynamics,
  author    = {JunYoung Park and Gyuyeon Lim and Myung-Kyu Yi},
  title     = {Dynamics-Aware Action State Flow for Wearable Human Activity Recognition},
  journal   = {IEEE Sensors Journal},
  volume    = {26},
  number    = {9},
  pages     = {13557--13572},
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/JSEN.2026.3675557}
}
```

## Contact

For questions or issues, please contact:
- JunYoung Park : park91802@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
