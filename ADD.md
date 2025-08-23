# Transient Detector Algorithm Description Document (ADD)
*Warning: this document is AI slop.*
## 1. Introduction
The transient detector described herein is a machine learning-optimized algorithm designed for the detection of transient events in audio signals, with particular emphasis on musical instrument recordings such as electric guitar, bass, and percussion. The system processes mono audio data sampled at 48 kHz and segments the input into fixed-length chunks for both training and evaluation. Its primary purpose is to provide a robust, differentiable, and tunable transient detection solution suitable for musical audio analysis and editing tasks, enabling automated identification of transient events such as note onsets and drum hits for downstream applications including audio segmentation, feature extraction, and intelligent editing.

This algorithm is applicable to a range of use cases, including guitar and bass audio processing, percussion and drum transient detection, audio editing and segmentation, and machine learning research in audio event detection. The design philosophy integrates conventional signal processing techniques—such as filtering, envelope detection, and dynamic compression—with learnable parameters that are optimized using global optimization methods. All core signal processing operations are implemented in JAX, ensuring differentiability and efficient batch processing, which facilitates both gradient-based and population-based optimization strategies. The architecture is modular, supporting optional multi-band filtering and dynamic compression, and allows for a variable number of processing channels to accommodate different experimental configurations.

Key innovations of this system include the use of learnable filter parameters (center frequency and Q factor) for each channel, learnable envelope detection parameters (moving average window size and channel weights), and optional dynamic compression with a learnable window and gain. The implementation is fully differentiable and jitable using JAX, enabling efficient training and evaluation. Furthermore, the architecture is sufficiently flexible to support ablation studies and hyperparameter sweeps, making it suitable for both research and production environments.
## 2. Theoretical Foundation
The theoretical foundation of the transient detector is rooted in the mathematical characterization of audio transients and the application of established digital signal processing techniques. In the context of musical audio, a transient is defined as a short-duration, high-energy event that typically marks the onset of a note or percussive hit. These events are distinguished by rapid changes in amplitude and spectral content, and their accurate detection is essential for tasks such as onset detection, segmentation, and feature extraction.

The algorithm leverages several core signal processing concepts to achieve robust transient detection. Envelope following is employed to track the energy profile of the audio signal over time, enabling the identification of sudden increases that correspond to transient events. Bandpass filtering is optionally applied to isolate specific frequency bands, enhancing the detector's sensitivity to transients within targeted spectral regions. Dynamic compression may also be utilized to attenuate sustained signal components, thereby emphasizing transient features.

All signal processing operations are implemented in a differentiable manner using the JAX library. This approach allows the entire detection pipeline—including filtering, envelope extraction, and channel combination—to be optimized via machine learning techniques. Differentiability ensures that the parameters governing each stage of the algorithm can be efficiently tuned using gradient-based or population-based optimization methods, facilitating the development of a highly adaptive and performant transient detection system.
## 3. Algorithm Architecture
The architecture of the transient detector is organized as a modular signal processing pipeline, designed to efficiently process audio data and extract transient events. The system operates on mono audio sampled at a fixed rate, segmenting the input into manageable chunks for analysis. Each chunk is processed through a series of stages, which may be selectively enabled or disabled according to the chosen hyperparameters.

The processing flow begins with the optional application of dynamic compression, which serves to attenuate sustained signal components and accentuate transient features. Following compression, the audio signal is distributed across multiple parallel processing channels. Each channel may optionally apply a bandpass filter, with learnable center frequency and Q factor, to focus on specific spectral regions relevant to transient detection.

Within each channel, the filtered (or unfiltered) signal undergoes envelope extraction, wherein the energy profile is computed using a moving average of the squared amplitude. The window size for this moving average is a learnable parameter, allowing the system to adapt its sensitivity to the temporal characteristics of transients. The resulting envelope is then scaled by a channel-specific weight, also subject to optimization.

The outputs of all channels are linearly combined, and a bias term is added to form the pre-activation signal. This pre-activation is further transformed by a sigmoid nonlinearity, with additional post-gain and post-bias parameters, to produce the final detection output. Thresholding is applied to this output to identify the precise timing of transient events.

The architecture supports flexible configuration, including the number of processing channels, the use of filters and compression, and the adjustment of all relevant parameters. This modularity enables comprehensive experimentation and optimization, ensuring that the system can be tailored to a wide range of audio analysis tasks.

# 4. Core Processing Components
## 4.1 Dynamic Compression
Dynamic compression is an optional stage in the transient detector pipeline, designed to attenuate sustained signal components and thereby emphasize transient features. The compressor operates by calculating a moving average of the squared audio signal, which serves as an estimate of the local signal power. Let $x[n]$ denote the input audio signal and $w_c$ the compressor window size in seconds. The compressor envelope $e_c[n]$ is computed as:

$$
e_c[n] = \sqrt{\frac{1}{N_c} \sum_{k=0}^{N_c-1} x^2[n-k]}
$$

where $N_c$ is the number of samples in the window, determined by $w_c$ and the sample rate. The compressed signal $y[n]$ is then obtained by modulating the input with the compressor gain $g_c$:

$$
y[n] = x[n] \cdot (1 - g_c \cdot e_c[n]) + \epsilon
$$

where $g_c$ is a learnable gain parameter and $\epsilon$ is a small constant to prevent numerical instability. Both the window size and gain are subject to optimization during training.
## 4.2 Multi-band Filtering
Multi-band filtering is optionally applied at the beginning of each processing channel to isolate specific frequency bands relevant to transient detection. Each channel may employ a bandpass biquad filter, parameterized by a learnable center frequency $f_0$ and quality factor $Q$. The filter coefficients are computed using standard digital filter design equations, and the filtering operation can be expressed as:

$$
z[n] = \text{Biquad}(x[n]; f_0, Q)
$$

where $z[n]$ is the filtered output. For differentiable training, the biquad filter may be approximated by a finite impulse response (FIR) filter, allowing gradients to propagate through the filtering operation. The parameters $f_0$ and $Q$ are optimized to maximize transient detection performance.
## 4.3 Envelope Detection
Envelope detection is a critical stage in the transient detector, responsible for extracting the energy profile of the signal within each channel. The envelope is computed by first squaring the filtered signal to obtain the instantaneous power, followed by a causal moving average with a learnable window size $w_e$. For a signal $z[n]$, the envelope $e[n]$ is given by:

$$
e[n] = \sqrt{\frac{1}{N_e} \sum_{k=0}^{N_e-1} z^2[n-k]}
$$

where $N_e$ is the number of samples in the envelope window. The moving average kernel is designed to be differentiable, with kernel weights smoothly dependent on $w_e$ during training. At inference time, a fixed rectangular window is used for efficient computation. This approach allows the system to adapt its sensitivity to the temporal characteristics of transients.
## 4.4 Channel Combination
The outputs of all processing channels are combined to produce the final transient detection signal. Each channel output is scaled by a learnable weight, and the weighted sum is augmented by a bias term to form the pre-activation signal $a[n]$:

$$
a[n] = \sum_{i=1}^{C} w_i e_i[n] + b
$$

where $C$ is the number of channels, $w_i$ are the channel weights, $e_i[n]$ are the channel envelopes, and $b$ is the bias. The pre-activation is then transformed by a sigmoid nonlinearity, with additional post-gain $g_p$ and post-bias $b_p$ parameters, yielding the final output $s[n]$:

$$
s[n] = \sigma(g_p a[n] + b_p)
$$

where $\sigma(\cdot)$ denotes the sigmoid function. All combination parameters are learnable and subject to optimization during training.
# 5. Learning Algorithm
## 5.1 Optimization Approach
The learning algorithm for the transient detector is designed to optimize all learnable parameters governing the signal processing pipeline, including filter coefficients, envelope window sizes, channel weights, and bias terms. Optimization is performed using global search methods, specifically differential evolution and basin hopping, which are well-suited for non-convex, high-dimensional parameter spaces. These methods operate by iteratively exploring candidate solutions and selecting those that minimize the loss function over the training dataset.

Parameter bounds are strictly enforced during optimization to ensure physical plausibility and numerical stability. For example, window sizes are constrained to positive values within a reasonable range, filter frequencies are limited to the audio band, and weights and gains are bounded to prevent excessive amplification or attenuation. Hyperparameter selection, such as the number of channels and the choice of enabled components (filters, compression), has a significant impact on the optimization outcome and is determined through systematic experimentation.
## 5.2 Loss Function
The objective of the optimization process is to minimize the discrepancy between the model's predictions and the ground truth transient labels. This is quantified using the mean squared error (MSE) loss function. Let $\hat{y}_i[n]$ denote the predicted output for chunk $i$ and $y_i[n]$ the corresponding ground truth label. The loss for a batch of $M$ chunks, each of length $N$, is given by:

$$
\mathcal{L} = \frac{1}{M N} \sum_{i=1}^{M} \sum_{n=1}^{N} (\hat{y}_i[n] - y_i[n])^2
$$

Batched processing is employed to efficiently compute the loss over multiple audio chunks, leveraging JAX's vectorization capabilities to parallelize the computation across the optimization population. This approach enables rapid evaluation of candidate parameter sets and accelerates the convergence of the optimization algorithm.
## 5.3 Training Procedure
Training begins with the initialization of all learnable parameters, either from default values or randomized within their respective bounds. The optimization loop proceeds by generating a population of candidate parameter sets, evaluating their performance using the loss function, and updating the population according to the rules of the chosen global optimization method. Differential evolution, for example, employs mutation and crossover operations to explore the parameter space, while basin hopping alternates between local minimization and stochastic jumps.

Convergence is assessed based on the stability of the loss function and the improvement of detection metrics on the training and validation datasets. The optimization process is terminated when further iterations yield negligible improvement, or when a predefined maximum number of iterations is reached. The final parameter set is selected based on its performance on the validation data, ensuring generalization and robustness of the transient detector.
# 6. Evaluation Methodology
## 6.1 Performance Metrics
The evaluation of the transient detector is based on standard metrics for event detection, including precision, recall, and F1-score. Detection accuracy is determined by comparing the predicted transient times to the ground truth annotations, with a match considered valid if the predicted event occurs within a specified temporal tolerance of the true event. Let $TP$, $FP$, and $FN$ denote the number of true positives, false positives, and false negatives, respectively. The metrics are defined as follows:

$$
	\text{Precision} = \frac{TP}{TP + FP}
$$

$$
	\text{Recall} = \frac{TP}{TP + FN}
$$

$$
	\text{F1 score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

Threshold selection plays a critical role in determining detection performance, as the output of the detector is a continuous signal that must be binarized to identify discrete transient events. The optimal threshold is chosen to maximize the F1-score on the validation dataset. Temporal matching is performed using a tolerance window, typically on the order of tens of milliseconds, to account for annotation uncertainty and perceptual relevance.
## 6.2 Evaluation Pipeline
The evaluation pipeline begins with the construction of the dataset, wherein audio recordings are segmented into fixed-length chunks and transient labels are generated based on annotated event times. The dataset is divided into training and validation subsets to enable cross-validation and assess generalization performance.

For each chunk, the model produces a sequence of detection scores, which are thresholded to yield predicted transient times. These predictions are matched to ground truth events using the defined temporal tolerance, and the counts of true positives, false positives, and false negatives are accumulated across all chunks. The evaluation metrics are then computed as described above.

This process is repeated for multiple threshold values to characterize the trade-off between precision and recall, and to identify the threshold that yields the best overall performance. The evaluation results are reported for both the training and validation datasets, providing a comprehensive assessment of the detector's accuracy and robustness.
# 7. Implementation Details
## 7.1 JAX-Specific Considerations
The implementation of the transient detector leverages the JAX library to enable efficient, differentiable computation throughout the signal processing pipeline. Just-In-Time (JIT) compilation is employed to transform core functions into highly optimized machine code, significantly accelerating both training and inference. Batched processing is facilitated by JAX's vectorization primitives, allowing the simultaneous evaluation of multiple audio chunks and candidate parameter sets. This capability is essential for population-based optimization algorithms and large-scale experimentation.

Gradient computation is fully supported for all differentiable operations, enabling the use of gradient-based optimization methods if desired. The design ensures that all learnable parameters, including filter coefficients, window sizes, and channel weights, are compatible with JAX's automatic differentiation framework. This approach provides flexibility in the choice of optimization strategy and supports rapid prototyping and experimentation.
## 7.2 Performance Optimization
Performance optimization is a key consideration in the implementation of the transient detector. GPU acceleration is utilized where available, with explicit device placement to ensure that large batches of audio data and model parameters are processed efficiently. Compilation caching is configured to minimize redundant compilation overhead, further improving runtime performance during repeated training and evaluation cycles.

Efficient batch handling is achieved through careful management of data structures and memory allocation, allowing the system to scale to large datasets and complex model configurations. These optimizations collectively enable the rapid exploration of the parameter space and support the practical deployment of the algorithm in both research and production environments.
## 7.3 Hyperparameter Variations
The transient detector architecture supports a wide range of hyperparameter configurations, enabling detailed ablation studies and performance analysis. The number of processing channels can be varied to assess the impact on detection accuracy and computational efficiency. Enabling or disabling bandpass filtering allows investigation of the role of spectral selectivity in transient detection, while the use of dynamic compression can be evaluated for its effect on the emphasis of transient features.

Systematic variation of these hyperparameters provides insight into the contributions of individual components and guides the selection of optimal configurations for specific audio analysis tasks. The flexibility of the implementation ensures that the algorithm can be tailored to diverse application requirements and experimental objectives.
# 8. Experimental Results
## 8.1 Performance Analysis
- Best Configuration: Optimal hyperparameter settings
- Ablation Studies: Contribution of individual components
- Comparative Analysis: Performance across different configurations
## 8.2 Visualization
- Time-domain Analysis: Signal and detection visualizations
- Channel Output Analysis: Individual channel contribution analysis
- Parameter Interpretation: Understanding optimized parameter values
# 9. Appendices
## 9.1 Mathematical Derivations
- Filter Design Equations: Detailed derivation of filter coefficients
- Moving Average Kernel: Mathematical foundation of differentiable kernel
## 9.2 References