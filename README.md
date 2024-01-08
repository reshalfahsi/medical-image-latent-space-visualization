# Medical Image Latent Space Visualization Using VQ-VAE


 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/medical-image-latent-space-visualization/blob/master/Medical_Image_Latent_Space_Visualization_Using_VQ-VAE.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


In this project, VQ-VAE (Vector Quantized VAE) is leveraged to learn the latent representation _z_ of various medical image datasets _x_ from MedMNIST. Similar to VAE (Variational Autoencoder), VQ-VAE consists of an encoder _q_(_z_|_x_) and a decoder _p_(_x_|_z_). But unlike VAE, which generally uses the Gaussian reparameterization trick, VQ-VAE utilizes vector quantization to sample the latent representation _z_ ~ _q_(_z_|_x_). Using vector quantization, it allows VQ-VAE to replace a generated latent variable from the encoder with a learned embedding from a codebook __C__ ∈ R<sup>_E_ × _D_</sup>, where E is the number of embeddings and _D_ is the number of latent variable dimensions (or channels in the context of image data). Let __X__ ∈ R<sup>_H_ × _W_ × _D_</sup> be the output feature map of the encoder, where _H_ is the height and _W_ is the width. To transform the raw latent variable to the discretized one, first we need to find the Euclidean distance between __X__ and __C__. This step is essential to determine the closest representation of the raw latent variable to the embedding. The computation of this step is roughly expressed as: (__X__)<sup>2</sup> + (__C__)<sup>2</sup> - 2 × (__X__ × __C__). This calculation yields __Z__ ∈ R<sup>_H_ × _W_</sup>, where each element denotes the index of the nearest embedding of the corresponding latent variable. Then, __Z__ is subject to __C__ to get the final discrete representation. Inspired by the centroid update of K-means clustering, EMA (exponential moving average) is applied during training, which updates in an online fashion involving embeddings in the codebook and the estimated number of members in a cluster.


## Experiment


To discern the latent space, go to [here](https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/Medical_Image_Latent_Space_Visualization_Using_VQ-VAE.ipynb).


## Result


## Evaluation Metric Curve

<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss of the model at the training stage. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/assets/mae_curve.png" alt="mae_curve" > <br /> MAE on the training and validation sets. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/assets/psnr_curve.png" alt="psnr_curve" > <br /> PSNR on the training and validation sets. </p>
<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/assets/ssim_curve.png" alt="ssim_curve" > <br /> SSIM on the training and validation sets. </p>


## Qualitative Result

Here is the visualization of the latent space:

<p align="center"> <img src="https://github.com/reshalfahsi/medical-image-latent-space-visualization/blob/master/assets/latent_space.png" alt="qualitative_result" > <br /> The latent space of five distinct datasets, i.e., DermaMNSIT, PneumoniaMNIST, RetinaMNIST, BreastMNIST, and BloodMNIST.</p>


## Credit

- [Neural Discrete Representation Learning](https://arxiv.org/pdf/1711.00937.pdf)
- [Vector-Quantized Contrastive Predictive Coding](https://github.com/bshall/VectorQuantizedCPC)
- [Variational AutoEncoder](https://keras.io/examples/generative/vae/)
- [Vector-Quantized Variational Autoencoders](https://keras.io/examples/generative/vq_vae/)
- [MedMNIST](https://medmnist.com/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
