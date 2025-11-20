# Extracted from 1.pdf

*Total pages: 48*

---

## Page 1

Stockformer: A Price-Volume Factor Stock Selection Model Based on
Wavelet Transform and Multi-Task Self-Attention Networks
Bohan Maa,1, Yushan Xuea,1, Yuan Lua, Jing Chena,∗
aSchool of Statistics and Mathematics, Central University of Finance and Economics, Beijing 100081, China
Abstract
As the Chinese stock market continues to evolve and its market structure grows increasingly com-
plex, traditional quantitative trading methods are facing escalating challenges. Particularly, due
to policy uncertainty and the frequent market fluctuations triggered by sudden economic events,
existing models often struggle to accurately predict market dynamics. To address these challenges,
this paper introduces “Stockformer”, a price-volume factor stock selection model that integrates
wavelet transformation and a multitask self-attention network, aimed at enhancing responsive-
ness and predictive accuracy regarding market instabilities. Through discrete wavelet transform,
Stockformer decomposes stock returns into high and low frequencies, meticulously capturing long-
term market trends and short-term fluctuations, including abrupt events. Moreover, the model
incorporates a Dual-Frequency Spatiotemporal Encoder and graph embedding techniques to effec-
tively capture complex temporal and spatial relationships among stocks. Employing a multitask
learning strategy, it simultaneously predicts stock returns and directional trends. Experimen-
tal results show that Stockformer outperforms existing advanced methods on multiple real stock
market datasets. In strategy backtesting, Stockformer consistently demonstrates exceptional sta-
bility and reliability across market conditions—whether rising, falling, or fluctuating—particularly
maintaining high performance during downturns or volatile periods, indicating a high adaptability
to market fluctuations. To foster innovation and collaboration in the financial analysis sector,
the Stockformer model’s code has been open-sourced and is available on the GitHub repository:
https://github.com/Eric991005/Multitask-Stockformer .
Keywords: Price-Volume Factor Selection, Stockformer, Wavelet Transform, Spatiotemporal
Graph Embedding, TopK-Dropout Strategy
Preprint submitted to Expert Systems With Applications June 18, 2024arXiv:2401.06139v2  [q-fin.TR]  17 Jun 2024

## Page 2

1. Introduction
Forecasting stock returns has long been a widely researched topic in the field of finance. While
the classical efficient market hypothesis posits that stock market prices, based on publicly available
information, are unpredictable (Schwartz, 1970), more recent studies indicate that variables such
as interest rates, inflation, and investor sentiment can significantly predict future stock market
returns (Bollerslev, Marrone, Xu & Zhou, 2014; Ang & Bekaert, 2007; Campbell & Thompson,
2008). Beyond market returns, the ability to predict cross-sectional individual stock returns—which
has identified over 400 stock characteristics, humorously termed the ”factor zoo”—raises questions
about the extent to which individual stock returns can be predicted and which stock characteristics
truly provide useful information for out-of-sample return prediction (Cochrane, 2011; Harvey, Liu
& Zhu, 2016; Green, Hand & Zhang, 2017). Exploring these questions within the Chinese capital
market, which has an A-share market capitalization of RMB 68 trillion, is crucial for improving
the effective allocation of these vast financial resources.
The challenge of predicting out-of-sample stock returns in China arises from several issues.
First, the multitude of factors affecting stock returns and the low signal-to-noise ratio in high-
dimensional sparse matrices make it difficult for traditional econometric models to extract mean-
ingful information. Second, the functional relationships between predictive features and stock
returns are uncertain (Campbell & Cochrane, 1999; He & Krishnamurthy, 2013), posing challenges
in capturing the nonlinear structures between them. Third, given its relatively short history of
just over two decades and the continuous evolution of its regulatory framework, the Chinese stock
market is particularly susceptible to policy shifts and other abrupt events that significantly impact
the market (Li, Ma, Zhang & Zhang, 2020; Wang, Tsai & Li, 2017).
In recent years, deep learning has become an indispensable tool in quantitative investment,
particularly in enhancing multifactor strategies that form the basis for understanding stock price
movements (Wang, Zhuang & Feng, 2022). By automating feature learning and capturing non-
linear relationships in stock market data, deep learning algorithms effectively identify complex
patterns, thus enhancing prediction accuracy (Guo, Wang, Ni & Shum, 2022). While the global
research community recognizes the potential of deep neural networks, such as Recurrent Neural
∗Corresponding author
Email address: chenjingma@cufe.edu.cn (Jing Chen)
1Bohan Ma and Yushan Xue contributed equally to this work.
2

## Page 3

Networks (RNNs) and Convolutional Neural Networks (CNNs), for predicting stock and futures
prices, preliminary studies have shown promising results (Senxin, Sitong, Linna et al., 2023; Yue,
Liu & Zhang, 2022). However, the use of deep learning models like RNNs and CNNs, although
extensive, seldom explores deeper neural network models that mine and construct market and yield
sequence information, suggesting room for further development in the application of deep learning
to stock markets.
This paper proposes a multitask prediction model for stocks based on wavelet transformation
and self-attention networks. The main contributions of this work are:
1.Multifactor Model Input: We constructed 360 price-volume factors as the primary feature
inputs for the Stockformer model. These factors, rigorously tested statistically, cover multiple
dimensions such as price, volume, and volatility, ensuring comprehensive market information
capture.
2.High and Low Frequency Decomposition: Discrete wavelet transformation is used to
decompose returns into high and low frequencies. This technique allows the model to cap-
ture both short-term market fluctuations (including abrupt events) and long-term trends,
enhancing detail detection in market dynamics.
3.Spatiotemporal Encoder and Attention Mechanism: A dual-channel spatiotemporal
encoder integrated with a fusion attention mechanism precisely predicts the returns and
trend movements of individual stocks. This method handles both temporal dependencies
and spatial correlations among stocks, thereby improving prediction accuracy and efficiency.
4.Graph Embedding Techniques: By constructing spatial relationship graphs and tempo-
ral graphs of stocks, graph embedding methods are employed to deeply analyze the spatial
and temporal characteristics of stocks, aiding the model in capturing dynamic market rela-
tionships.
5.Multitask Learning Strategy: The multitask learning approach not only predicts the
returns of stocks but also their upward or downward trends. This multi-output framework
enables the model to provide more comprehensive market change predictions, enhancing the
robustness of the forecasts.
3

## Page 4

2. Related Work
2.1. Factor Construction
A factor is a quantifiable variable that influences stock returns, encapsulating the underlying
economic and financial dynamics. The conventional factors encompass market, style, industry,
macroeconomic, price-volume, and machine learning-derived factors. The seminal works by Fama
et al. (Fama & French, 1992, 2015) introduced market factors such as excess returns and volatility,
encapsulating the overall market performance and risk. Green et al. (Green et al., 2017) utilized
cross-sectional regression on 94 US stocks to investigate style factors, shedding light on the relative
performance of diverse stock categories. More recent work by Fan et al. (Fan, Liao & Wang, 2016)
explores time-varying factor loadings, enhancing the adaptability of factor models to changing
market conditions. Furthermore, Jensen et al. (Jensen, Kelly & Pedersen, 2021) discusses the
stability of beta-adjusted equity factors, confirming the presence of seasonal and momentum effects
in the cross-section of factor returns.
Our research focuses on utilizing the price-volume factor of Alpha360 to optimize the inputs
of deep learning models and improve the prediction performance.The Alpha360 dataset includes
daily data on six fundamental stocks, which is used to form comprehensive data inputs through
historical backtracking and feature construction to enhance the model’s ability to capture market
dynamics.
2.2. Stock Return Prediction Models
Recent breakthroughs in machine learning in dimensionality reduction, penalty terms, and
functional techniques naturally excel in addressing the challenges mentioned in the Introduction
extracting effective information and dealing with non-linear relationships in stock market data. Re-
cent papers have explored various types of machine learning algorithms for predicting stock returns.
The first category includes commonly used dimensionality reduction models in finance. These
models compress high-dimensional data into lower dimensions while preserving key information.
For instance, Ampomah et al. (Ampomah, Qin & Nyame, 2020) and Qolipour et al. (Qolipour,
Ghasemzadeh & Mohammad-Karimi, 2021) employed Principal Component Analysis (PCA) to
simplify sets of fundamental features and technical indicators, and integrated PCA with tree-based
machine learning classifiers to predict stock returns and price movements. Moreover, Zhong et al.
4

## Page 5

enhanced the accuracy of stock return predictions significantly by applying various dimensionality
reduction techniques such as PCA, Fuzzy Robust PCA (FRPCA), and Kernel-based PCA (KPCA),
demonstrating the effectiveness and potential of reduction techniques in handling complex financial
datasets. The second category encompasses linear models with penalty terms, which excel by
incorporating penalty terms to reduce noise information load, thus enhancing prediction accuracy.
Such models have shown excellent performance in the financial domain. For example, Yacine et al.
(A¨ ıt-Sahalia, Fan, Xue & Zhou, 2022) explored the predictability of high-frequency stock returns
using methods like LASSO. Xuemei et al. (Xuemei & Junwen, 2024) proposed a multi-logistic re-
gression model combined with penalty terms such as G-LASSO, G-SCAD, and G-MCP, effectively
improving the predictive power for stock return direction in high-dimensional data settings. The
third category includes nonlinear models, which can fit the nonlinear structure between predictive
variables and returns based on historical data. Scholars using artificial intelligence algorithms like
Random Forest, Fuzzy Neural Networks, and Long Short-Term Memory (LSTM) networks have
tested the effects of technical and macroeconomic predictive factors on daily stock price returns’
explained variance (Fischer & Krauss, 2018; Sirignano, Sadhwani & Giesecke, 2016; Bao, Yue &
Rao, 2017; Butaru, Chen, Clark, Das, Lo & Siddique, 2016). With continuous advancements in
computing technology, deep learning has been widely applied in quantitative trading, particularly
in predicting stock returns, as shown by Chong et al. (Chong, Han & Park, 2017), who utilized
an Autoencoder (AE) to transform raw data for use in Deep Neural Networks (DNNs), predicting
future returns for 38 stocks in the Korean market. Dami et al. (Dami & Esterabi, 2021) combined
AE and LSTM models to optimize stock return predictions for ten companies on the Tehran Stock
Exchange. Additionally, Gunduz (Gunduz, 2021) used a Variational Autoencoder (VAE) to predict
the hourly direction for eight banks in the BIST 30 index, demonstrating accuracy comparable to
models trained with un-reduced features. Due to these advantages, machine learning technologies
have become a frontier application in the financial domain, particularly in predicting financial
market movements, processing textual information, and improving trading strategies.
The Chinese stock market remains in a phase of continuous development and improvement,
and economic policy uncertainty (Baker, Bloom & Davis, 2016) renders the predictability of stock
returns challenging. Numerous scholars have integrated machine learning and deep learning tech-
nologies to address the prediction of expected returns in the Chinese stock market. Jiang et
5

## Page 6

al.(Jiang, 2011) investigated the predictability of stock returns for portfolios categorized by in-
dustry, size, price-to-book ratios, and ownership concentration; Li et al. (Li, Bu & Wu, 2017)
found that deep learning outperformed traditional econometric models in predicting the CSI 300
index. Moreover, considering the dynamic changes in stock market styles (concept drift), Song et
al. (Song, Xiao, Zhang & Xia, 2023) proposed a Contextual Information Shift Perception (CISP)
method that incorporates dynamic parameters in stock predictions, effectively adapting to changes
in the Chinese market style and thus providing more accurate forecasts. Although economic policy
uncertainty and its resultant abrupt events are widely recognized as key factors influencing stock
market volatility (Cai, Zhang, Han, Liang et al., 2022; Wang, Zhang, Zhang, Gao & Lin, 2021; Li &
Huang, 2021), current market models often fail to identify the specific timing and impact of these
events. Furthermore, existing models frequently do not adjust their predictions promptly after
abrupt events occur, leading to prediction biases due to excessive reliance on factor momentum ef-
fects (Ehsani & Linnainmaa, 2022). This phenomenon not only weakens the models’ responsiveness
to rapid market changes but also limits their practical application in trading.
This research introduces a model that utilizes advanced signal processing techniques—wavelet
transformation—to identify and isolate the signals of abrupt events within stock return sequences.
Furthermore, by incorporating a self-attention mechanism, the model ensures rapid adaptation
to market changes following such events, thereby enhancing the accuracy and timeliness of its
predictions.
2.3. Backtesting Trading Strategy
Backtesting trading strategies is an integral part of the investment process, enabling the assess-
ment of strategy performance and risk management before real-world application. Fung and Hsieh
(Fung & Hsieh, 1997) provide an empirical examination of hedge funds’ dynamic trading strategies,
establishing a framework for backtesting and evaluation. Georgakopoulos (Georgakopoulos, 2015)
demonstrates the utility of automated programming for backtesting, detailing processes from data
acquisition to strategy implementation and risk assessment. Chan (Chan, 2021) offers a thorough
overview of quantitative trading foundations, with a focus on backtesting, data handling, and strat-
egy execution. De Prado’s work (De Prado, 2018) highlights the convergence of machine learning
with financial backtesting, introducing ”out-of-sample” testing to enhance strategy robustness.
Qian (Qian, Hua & Sorensen, 2007) discusses quantitative equity portfolio management, covering
6

## Page 7

backtesting and optimization across strategy development and performance evaluation.Together,
these contributions provide theoretical frameworks, practical insights, and extensive case studies
for backtesting trading strategies. Despite these advancements, there is still unexplored territory
in backtesting, suggesting room for further research and innovation in strategy development and
validation.
This study conducts backtesting using the Qlib framework developed by Microsoft2, employing
the TopK-Dropout trading strategy, and compares it with traditional and advanced models to
validate the effectiveness and stability of the Stockformer model. This approach not only enhances
the transparency of the strategy but also ensures its stability across different market conditions.
3. Preliminaries
3.1. Problem Definition
The definition of stock prediction problems varies depending on the specific investment strategy
adopted by investors. In this paper, we employ a widely accepted paradigm in the research field,
the panel data analysis method (Muhammad & Ali, 2018). This approach integrates the returns,
trend movements, and 360 price-volume factors of multiple stocks over a past period to predict
the future returns and trend movements of multiple stocks for an upcoming period. This method
fully utilizes multidimensional data resources and provides a theoretical foundation and practical
guidance for building robust investment portfolios and exploring potential arbitrage opportunities
by considering various financial factors, including both fundamental and technical aspects.
Consider the task of predicting the returns and trend movements of multiple stocks using
the Stockformer method. Let xi
t∈R362represent the feature vector of stock characteristics
for the i-th stock at time step t, consisting of the return rate (1 dimension), trend direction
(1 dimension), and the 360 dimensions from the Alpha360 factor library, where i∈ {1, . . . , N },
andNis the number of stocks considered. The feature tensor of all Nstocks at time step t
is denoted as Xt=
x1
t, . . . , xi
t, . . . , xN
tT∈RN×362. Given a historical dataset of stock fea-
tures: X={X1, . . . , X T1} ∈RT1×N×362, where T1is the number of historical time steps, the
objective is to predict the returns (1 dimension) and trend movements (0 for a downtrend, 1
for an uptrend, 1 dimension) for the subsequent T2time steps. The predicted stock indicators
2Qlib: https://github.com/microsoft/qlib
7

## Page 8

are denoted as ˆY=n
ˆYT1+1, . . . , ˆYT1+T2o
∈RT2×N×2, where ˆYt∈RN×2represents the pre-
dicted stock indicators at time step t. The true values of these predictions are represented by
Ytrue={YT1+1, . . . , Y T1+T2} ∈RT2×N×2.
The challenge lies in accurately forecasting ˆYbased on the patterns and trends identified in X,
while considering various factors that influence stock dynamics.
3.2. Self-Attention Mechanism
The self-attention mechanism(Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser &
Polosukhin, 2017) is the most commonly used attention mechanism that improves the accuracy of
stock return forecasts by allowing the model to capture global dependencies across the entire input
sequence. The mechanism processes inputs composed of queries, keys, and values, each having a
dimension d. The procedure involves computing the dot products of the query with all the keys,
dividing each by√
d, and then applying a softmax function to obtain the weights on the values.
Mathematically, this operation can be represented as:
Att(Q, K, V ) = softmax(QWQ)(KTWK)√
d
(V WV), (1)
where WQ,WK, and WVare the learnable parameters of the projections, and Att( ·) denotes the
self-attention operation.
3.3. Wavelet Transform
Wavelet transform, introduced by Daubechies Daubechies (1992), is a mathematical tool used
for the hierarchical decomposition of signals. It involves scaling functions and wavelet functions,
which create a stable basis in the signal space through their shifts and expansions. This study
explores the effectiveness of discrete wavelet transform in financial time series such as the return
sequence of a single stock. For a one-dimensional stock return sequence rt∈RT, the discrete wavelet
transform is applied to decompose the return sequence into low and high frequency components:
rlk=X
jgj−2krj, r hk=X
jhj−2krj, (2)
where g={gk}k∈Zandh={hk}k∈Zare the low-pass and high-pass filters, respectively. The low-
frequency component rlcaptures the long-term trends of stock returns, while the high-frequency
component rhreflects short-term fluctuations and abrupt events. This decomposition aims to
provide new perspectives and analytical tools for understanding and predicting market dynamics.
8

## Page 9

4. Methodology
4.1. Overall Model Architecture
N stock return seriesHigh Pass Filter hLow Pass Filter lInverse High Pass FilterInverse Low Pass Filter
FCFC
PredictorPredictor
FC1
FC2
N * DT * DFC1FC2MAELossCross Entropy Loss
MAELossCross Entropy Loss
Struc2VecDecoupling Flow Layer
Temporal AttentionDilated Casual ConvolutionDual-Frequency Spatiotemporal EncoderDual-Frequency Fusion DecoderQuerySampleCopyFCFCFCFCFCFCFCFC
FC
FCFCScaleMaskSoftmaxScaleSoftmaxconcatconcatScaleMaskSoftmax
ScaleMaskSoftmax
Graph Positional EncodingTime Slot EmbeddingSpatial GraphTemporal GraphTime Slot and Struc2VecGraph AttentionFusion Attention
: Positional EncodingFC: Fully-connected Layer: Trend Indicators & Multifactor Feature Input: Temporal Embedding of High-dimensional Features: Spatial Embedding of High-dimensional Features
Figure 1: Stockformer architecture diagram, which primarily consists of three parts: the Decoupling Flow Layer, the
Dual-Frequency Spatiotemporal Encoder, and the Dual-Frequency Fusion Decoder.
Based on Figure 1, the Stockformer architecture is primarily divided into three parts: the
Decoupling Flow Layer , the Dual-Frequency Spatiotemporal Encoder , and the Dual-
Frequency Fusion Decoder . Initially, the historical stock data X ∈RT1×N×362is processed
through the Decoupling Flow Layer , where the stock return series tensor undergoes wavelet
transformation to separate into high andlow frequency components , while other parts (such
as trend indicators and price-volume factors) remain unchanged. These components are then
concatenated with the unchanged parts along the third dimension. Here, the low-frequency
component (denoted as l) captures long-term trends, and the high-frequency component
(denoted as h) captures short-term fluctuations and abrupt events. These are denoted as Xh,Xl∈
RT1×N×362, where Xh,Xlare linearly transformed through a fully connected layer to RT1×N×D.
Subsequently, a Dual-Frequency Spatiotemporal Encoder is designed to represent these
distinct time series patterns: the low-frequency features are fed into a Temporal Attention
(denoted as tatt) module, while the high-frequency features are processed through a Dilated
9

## Page 10

Causal Convolutional Layer (denoted as conv), represented as Xtatt
l,Xconv
h∈RT1×N×D. These
components are then input into Graph Attention Networks (denoted as gat), interacting with
graph information to enable the model to capture complex relationships and dependencies among
stocks and time. In this module, the spatial graph Aspaand temporal graph Atemare transformed
through a fully connected layer and tensor broadcasting operations to high-dimensional embeddings
denoted as ρspa, ρtem∈RT1×N×D, which are then fused with Xtatt
l,Xconv
hthrough addition and
undergo graph attention operations to produce Xgat
l,Xgat
h∈RT1×N×D. The Dual-Frequency
Spatiotemporal Encoder consists of Lstacked layers, aimed at effectively representing the dual-
scale spatiotemporal patterns of low and high-frequency waves. Finally, in the Dual-Frequency
Fusion Decoder , predictors generate ˆYf
i,ˆYf
h∈RT2×N×D, which are aggregated through Fusion
Attention interactions to obtain a latent representation of dual-scale temporal patterns ˆYf∈
RT2×N×D. Through distinct fully connected layers (regression layer FC1 and classification layer
FC2), multi-task outputs are produced, including stock return predictions (regression result,
denoted as reg)ˆYreg∈RT2×Nand stock trend prediction probabilities (classification result, denoted
ascla)ˆPcla∈RT2×N. Additionally, regression values for the low-frequency component ˆYlreg∈
RT2×Nand trend prediction probabilities ˆPlreg∈RT2×Nare output to enhance learning of low-
frequency signals in the supervision signal, with detailed loss function calculations presented in the
Multi-Supervision subsection 4.4.2.
4.2. Decoupling Flow Layer
Given a historical stock data set X, focusing on the stock return series, this study employs
Discrete Wavelet Transform (DWT) from the Decoupling Flow Layer to extract the low and high
frequency component sequences of returns, representing long-term and short-term temporal pat-
terns respectively. The low-frequency component, stable and indicative of long-term trends, reflects
the trajectory of returns; while the volatile high-frequency component reflects short-term fluctu-
ations, such as those induced by economic policy uncertainties leading to abrupt market events.
The DWT of the input return series Xcan be represented by Equation 3:
Xl=gX,Xh=hX, (3)
where the candidate low and high frequency components XlandXhundergo a downsampling
operation that reduces the time step length to half in DWT. Consequently, this layer uses the
10

## Page 11

inverse low-pass and high-pass filters gTandhTto upsample the inputs. The trend indicators
(0,1) and the 360-dimensional multifactor features (Alpha360) are then concatenated with the
upsampled low and high frequency components, forming the final multi-dimensional low and high
frequency inputs. These are then transformed by a fully connected layer into high-dimensional low
and high frequency components Xl,Xh, enhancing the representational power of the Stockformer.
The upsampling operation and fully connected layer are formalized in Equation 4:
Xl=WggTXl+bg,Xh=WhhTXh+bh, (4)
where Wg, Wh∈RC×dandbg, bh∈Rdare learnable parameters.
4.3. Dual-Frequency Spatiotemporal Encoder
The Dual-Frequency Spatiotemporal Encoder is comprised of three major components: Tem-
poral Attention, Dilated Causal Convolution, and Time Slot with Struc2Vec Graph Attention
Networks.
4.3.1. Decoupling Temporal Feature Extraction
Unlike previous works that solely relied on methods like LSTM for processing financial se-
quences, our approach integrates time convolution layers and time attention to focus on trends
and seasonal components. Time attention captures long-term low-frequency trends by considering
global sequence relationships, while the time convolution layer focuses on local patterns, effectively
simulating high-frequency components and abrupt events. This dual modeling approach enhances
the prediction accuracy of complex financial sequences. Dilated causal convolution is a specific
type of one-dimensional convolution that slides over the input by skipping values at a defined step,
as shown in Figure 1. Theoretically, given a one-dimensional sequence input x∈RTand a filter
f∈RJ, the dilated causal convolution operation at time step tis defined as:
x ⋆ f(t) =JX
j=0f(j)x(t−c×j), (5)
where cis the dilation factor. The dilated causal convolution of the high-frequency component is
expressed as:
Xconv
h= ReLU(Θ ⋆Xh+b) (6)
where Θ and bare learnable parameters, and ReLU( ·) is the rectified linear unit.
11

## Page 12

Here, self-attention is utilized to capture the long-term trends in the low-frequency component,
reflecting the stability and prominent long-term trends observed in return trajectories:
Xtatt
l= Concat( ta1, . . . , ta n, . . . , ta N),
where tan= Att( Xn
l, Xn
l, Xn
l)(7)
4.3.2. Time Slots and Struc2Vec Graph Attention Networks
Time Slots .As previously discussed, the stock returns exhibit cyclical fluctuations, with each
day in the dataset represented by a timestamp t, which is considered as a unit of time. Thus, it is
essential to extract time features from t. A straightforward approach is to treat each timestamp as a
floating-point number and then use a multilayer perceptron model to convert this value into a fixed-
length vector. However, this method has two disadvantages. Firstly, each timestamp is typically a
large integer, which could dominate other features when used directly. Secondly, there are monthly
and daily cyclic variations between different timestamps that cannot be captured merely by the
timestamps. For example, trading conditions during the same period across different months might
be similar. To address these issues, we employ a method of time slots to represent timestamps.
Given a baseline timestamp t0(to ensure t−t0≥0,t0must be less than any timestamp in the
training and testing data), and a time unit ∆ t, we can establish intervals such as
[t0, t0+ ∆t),
[t0+ ∆t, t0+ 2∆ t),
···
These intervals are referred to as time slots, each with a size of ∆ t, which represents one day in
this dataset.
For simplicity, each time slot is represented by its index number. For instance, [ t0, t0+ ∆t) is
represented as 0. A timestamp tcan be mapped to a specific time slot tp, where t≥t0, computed
as shown in Equation 8:
tp=t−t0
∆t
(8)
For a finer granularity of each timestamp, the remainder tris recorded to ensure the uniqueness of
t, where 0 ≤tr<∆t, calculated as shown in Equation 9:
tr=t−t0−tp∆t (9)
In summary, each timestamp tcan be represented as ⟨tp, tr⟩.
12

## Page 13

Constructing the Temporal Graph .Next, we explore how to capture temporal features by
embedding time slots. Given the nature of the dataset, each time slot can represent a day. Con-
sidering the periodic fluctuations in stock returns and the monthly cycle being the most common
cyclical fluctuation in economic cycles, it suffices to focus on all time slots within a month. With
approximately 250 trading days in a year distributed across months, each month averages 21 trad-
ing days. Inspired by Li, Fu, Wang, Shahabi, Ye & Liu (2018), we attempt to construct a temporal
graph to represent time slots and then apply graph embedding methods to initialize the time slot
embeddings. However, Li et al. (2018) constructed an undirected graph for time slots, failing
to capture the sequential relationship between them. Moreover, they overlooked the connection
between adjacent dates, thus missing daily periodicity.
Addressing the issue at hand, we adopt the time-slot method to represent time stamps, drawing
inspiration from Yuan, Li, Bao & Feng (2020), as shown in Figure 2. This representation is captured
by the graph G′=⟨V′, E′⟩. In this context, each node v′∈V′denotes a time slot. The edges in E′
are categorized into two types: (1) edges connecting adjacent date slots, symbolizing contiguous
time slots, thus providing smoothness; and (2) edges connecting adjacent months, denoting identical
time slots in consecutive months, ensuring similarity. As an illustration, this study designates ∆ tas
1 day, partitioning each month into 21 distinct time slots. Taking into account the twelve months
within a year, we construct a directed temporal graph with a size of 21 ×12 = 252. Ultimately, tp
can be mapped to a node v′∈V′, where the sequence number for v′is given by tp%252, with %
representing the modulus operator.
Embedding Time Slots in the Temporal Graph .Initially, one-hot encoding is used to rep-
resent each time slot Ot
i∈R|V′|, where |V′|is the total number of nodes in the temporal graph
G′. Then, a fully connected neural network (with weight matrix W∗t∈R|V′|×dt) is designed to
transform each one-hot encoded Ot
iinto a fixed-length dense vector Dt
i=W∗t⊤Ot
i. This embed-
ding is used as the initial value for Wt, resulting in a high-dimensional temporal graph embedding
ρtem∈RT1×D. Leveraging the broadcasting property, ρtemis replicated N times, ultimately
yielding ρtemwith dimensions ρtem∈RT1×N×D.
Struc2Vec Graph Embedding .By computing the Spearman correlation coefficients between
the returns data of Nstocks, we construct an N×Ncorrelation matrix to serve as the adjacency
matrix, which describes the connectivity between stocks. Each node represents a stock, and the
13

## Page 14

Feb. Jan. Mar. Apr. …… Nov. Dec.
0
1
19
20…… …….…………………………………………………………
/gid1171 /gid1171 ：Time SlotAdjacent day time 
slot boundariesAdjacent month 
time slot boundariesFigure 2: Temporal graph depicting a year with each time slot representing a day. Each red directed line connects
two adjacent time slots, while each black directed line connects the same time slot in two adjacent months.
weight of the edges reflects the correlation between stocks. In the Struc2Vec algorithm, each stock
node is initially randomly initialized with a vector. Through multiple iterations, each node’s vector
representation is refined by integrating information from neighboring nodes and the weighted neigh-
bor vectors using an attention mechanism. This process iterates until the vector representations
stabilize or the predetermined number of iterations is reached. Ultimately, after multiple rounds
of updates, each node’s (stock’s) vector representation will contain structural information about
its position in the graph and the influence of its neighbors, thereby achieving high-dimensional
stock relational graph embeddings ρspa∈RN×d. Leveraging the property of broadcasting, ρspais
replicated to match T1times, resulting in dimensions ρspa∈RT1×N×D.
Self-Attention Encoding .After obtaining the high dimensional temporal graph embedding and
the stock return association graph embedding, this study broadcasts and sums the two, concate-
nating them with the seasonal and trend components, respectively. This results in an input tensor
for self-attention encoding. The final time slot and Struc2Vec graph attention network can be
represented as in Equation 10:
Xgat= Concat ( sa1, . . . sa t, . . . , sa T1)
where sa at= Att
˜Xt,˜Xt,˜Xt
and ˜X=X+ρspa+ρtem(10)
14

## Page 15

Thus, through the time slot and Struc2Vec graph attention network mechanism, this study
obtains Xgat(The high-frequency signals are denoted as Xgat
hand the low-frequency signals as
Xgat
l, with both types undergoing the same processing steps here. ), a vector containing structural
information and localized graph characteristics. This attention mechanism is highly efficient and
expressive when handling graph data.
4.4. Dual-Frequency Fusion Decoder
In this study, predictors (i.e., fully connected layers) are applied on the temporal dimensions
of the representations encoded by the Dual-Frequency Spatiotemporal Encoder ( Xgat
landXgat
h)
to transform them into future representations of multi-step stock returns and trends, resulting
in future representations of low-frequency and high-frequency components ˆYf
iand ˆYf
h. Given
that high-frequency components often correspond to rapid changes or noise in the data, which
are challenging to predict and uncertain, and low-frequency components generally correspond to
long-term trends and global patterns, which are more persistent and stable, crucial for forecasting
overall trends and long-term changes. Therefore, through fusion attention and a multi-supervision
strategy, this paper integrates the information of low and high-frequency components, with par-
ticular emphasis on supervising the low-frequency components to extract useful long-term trend
information.
4.4.1. Decoupling Feature Fusion
As the aim of this study is not to predict low and high-frequency components but to forecast
future stock return sequences and trends based on these components and price-volume factors, as
shown in Figure 1, we further propose a fusion attention mechanism. This mechanism integrates
the representations of low-frequency and high-frequency components ˆYf
l,ˆYf
hinto the stock returns
ˆYfand captures future internal dependencies. Specifically, the fusion attention considers the low-
frequency component as the query, extracting useful long-term and short-term information from
both low and high-frequency components in two time attention mechanisms. The fusion attention
can be expressed as in Equation 11:
ˆYf= Concat ( fa1, . . . , fa n, . . . , fa N)
where fan= Att
ˆYfn
l,ˆYfn
l,ˆYfn
l
+ Att
ˆYfn
l,ˆYfn
h,ˆYfn
h
.(11)
15

## Page 16

In Equation 11, the study concatenates each stock n’s attention mechanism output fanto obtain
ˆYf. The attention mechanism calculates self-attention for low-frequency components and attention
between low-frequency and high-frequency components. This fusion attention mechanism allows
simultaneous utilization of both low and high-frequency component information, thereby better
capturing future stock returns and internal dependencies.
4.4.2. Multi-Supervision
During the training process, through Stockformer, the model is capable of making multi-
dimensional predictions on future trends of stock sequences. Specifically, the model’s output layer
is divided into classification and regression task outputs, respectively providing the probabilities
of stock trend predictions ˆPcla, low-frequency component trend predictions ˆPlcla, predicted stock
return values ˆYreg, and low-frequency component values ˆYlreg.
Loss Function for Multi-Supervision Optimization .As depicted in the Dual-Frequency
Fusion Decoder in Figure 1, the training integrates the loss functions of regression and classification
tasks, allowing the model to learn multiple types of outputs simultaneously. The loss function is
defined in Equation 12:
L=Lreg+λLcla (12)
where λis a hyperparameter used to balance the weights of regression and classification losses.
The computations for LregandLclaare as follows:
(1) Regression Loss Lreg:Used to measure the accuracy of the model in predicting the
returns and their low-frequency components, calculated using the Mean Absolute Error (MAE)
loss. Defined as shown in Equation 13:
Lreg=1
N(T2−T1)T1+T2X
t=T1+1NX
n=1 
|yn
t−ˆyn
t|+yn
lt−ˆyn
lt
(13)
where Nis the number of samples, ( T2−T1) is the number of time steps, yn
tis the actual return
of sample nat time t, ˆyn
tand ˆyn
ltare the predicted return and low-frequency component values,
respectively.
(2) Classification Loss Lcla:Used to assess the performance of the model in predicting stock
trends, typically using a cross-entropy loss function. Defined as shown in Equation 14:
Lcla=1
N(T2−T1)T2X
t=T1+1NX
n=1 
−X
kyn
c,t,klog 
ˆpn
cla,t,k
−X
kyn
c,t,klog 
ˆpn
lcla,t,k!
(14)
16

## Page 17

where yn
c,t,kis the true class label for sample nat time tfor class k. ˆpn
cla,t,kand ˆpn
lcla,t,kare
the predicted probabilities for the regular and low-frequency classification tasks, respectively. By
minimizing the above loss functions, the model can optimize the prediction of stock trends and
returns simultaneously, thereby enhancing the overall predictive performance and robustness of the
model.
5. Construction of Price-Volume Factors
5.1. Data Sources and Stock Pool Selection
The CSI 300 Index and the China Securities Index series are currently the mainstream indices
in the A-share market. The CSI 300 Index, consisting of the top 300 stocks in terms of liquidity
and market capitalization from both Shanghai and Shenzhen stock exchanges, effectively reflects
the overall trend of the A-share market and has good market representativeness. Therefore, this
paper selects the constituent stocks of the CSI 300 Index as the stock pool for constructing the
stock selection model.
Table 1: Stock Market Indicators
Market Indicator Open High Low Close VWAP Volume
Indicator Name open high low close vwap volume
This study analyzes the stock market data from March 1, 2018, to January 30, 2024. Consider-
ing potential adjustments such as dividends and stock splits, this paper applies forward adjustments
to the price-related indicators. Market indicators are presented in Table 1.
We utilize stock market data from March 1, 2018, to January 30, 2024, for analysis. As shown in
Figure 3, to maintain the model’s generalizability and adaptability over different periods, a rolling
window approach is used to generate sub-datasets. Considering the Chinese stock market is open
for about 243 days annually, the data is divided into two years of trading data as the training set
(486 days), followed by three months of trading data for the validation set (81 days), and another
three months for the test set (81 days), resulting in 14 sub-datasets. These are then split by time
into training (75%), validation (12.5%), and testing sets (12.5%) as detailed in Table 2.
Moreover, each subdataset underwent a filtering process for the stock pool selection:
17

## Page 18

4868181Training SetValidation SetTest SetPeriod 1
Start Date2018-03-01End Date2024-01-30Period 2Period 3Period 13Period 14Figure 3: Visualization of the dataset division method used in this study. Each segment represents a specific period
in the rolling window analysis, showing the division into training, validation, and test sets across the specified dates.
1. Stocks that were delisted, had negative net assets, were newly issued, or were under risk
warning were excluded.
2. Stocks that were suspended from trading or hit the price limit on the trading day were also
excluded.
5.2. Factor Construction
This study constructs price-volume factors for each subdataset by mining daily market data.
Utilizing the Alpha360 factor library developed by Microsoft’s Qlib framework, we constructed 360
price-volume factors, categorized into six types, with each type comprising 60 factors. Table 3
presents examples of factor construction methods for the first two factors in each category.
The function Refretrieves the historical value of a variable. For instance, in the Close/Close
category, the price-volume factors CLOSE0-CLOSE59 are calculated as the ratio of the close value
from 0 to 59 periods back to the current close value. The construction of other categories follows
a similar rationale. This study constructs these 360 price-volume factors daily for every stock in
the stock pool.
18

## Page 19

Table 2: Subdataset Division with Detailed Date Ranges
Training Set (486 days) Validation Set (81 days) Test Set (81 days)
Dataset Start Date End Date Days Start Date End Date Days Start Date End Date Days
Subdataset 1 2018-03-01 2020-02-28 486 2020-03-02 2020-06-30 81 2020-07-01 2020-10-29 81
Subdataset 2 2018-05-31 2020-05-29 486 2020-06-01 2020-09-23 81 2020-09-24 2021-01-25 81
Subdataset 3 2018-08-27 2020-08-26 486 2020-08-27 2020-12-25 81 2020-12-28 2021-04-28 81
Subdataset 4 2018-11-28 2020-11-27 486 2020-11-30 2021-03-30 81 2021-03-31 2021-07-28 81
Subdataset 5 2019-03-04 2021-03-02 486 2021-03-03 2021-06-30 81 2021-07-01 2021-11-01 81
Subdataset 6 2019-06-03 2021-06-01 486 2021-06-02 2021-09-27 81 2021-09-28 2022-01-26 81
Subdataset 7 2019-08-28 2021-08-26 486 2021-08-27 2021-12-28 81 2021-12-29 2022-05-05 81
Subdataset 8 2019-11-29 2021-11-30 486 2021-12-01 2022-03-31 81 2022-04-01 2022-08-01 81
Subdataset 9 2020-03-04 2022-03-03 486 2022-03-04 2022-07-04 81 2022-07-05 2022-11-02 81
Subdataset 10 2020-06-03 2022-06-06 486 2022-06-07 2022-09-28 81 2022-09-29 2023-02-03 81
Subdataset 11 2020-08-31 2022-08-30 486 2022-08-31 2022-12-29 81 2022-12-30 2023-05-05 81
Subdataset 12 2020-12-02 2022-12-01 486 2022-12-02 2023-04-03 81 2023-04-04 2023-08-02 81
Subdataset 13 2021-03-05 2023-03-06 486 2023-03-07 2023-07-05 81 2023-07-06 2023-11-03 81
Subdataset 14 2021-06-04 2023-06-05 486 2023-06-06 2023-09-28 81 2023-10-09 2024-01-30 81
5.3. Factor Neutralization and Factor Value Preprocessing
5.3.1. Factor Value Preprocessing
For each subdataset divided, factor values corresponding to each trading day are preprocessed,
which includes the following steps:
1.Missing Values: Fill missing factor values using the information from the previous trading
day of the stock.
2.Extreme Value Treatment: Exclude data points that fall beyond three standard devia-
tions from the mean, assuming a normal distribution.
3.Standardization: Apply Z-score normalization to each factor sequence to standardize the
values.
5.3.2. Neutralization Process
Industry and market value neutralization are employed to eliminate the influence of a stock’s
industry and market value. The process for each subdataset is as follows:
1. Convert each stock’s industry category into dummy variables.
19

## Page 20

Table 3: Construction Methods for Six Categories of Price-Volume Factors
Factor Category Factor Name Price-Volume Factor Construction Method
Close/Close CLOSE1 Ref(close ,1)/close
CLOSE2 Ref(close ,2)/close
. . . . . .
Open/Close OPEN1 Ref(open ,1)/close
OPEN2 Ref(open ,2)/close
. . . . . .
High/Close HIGH1 Ref(high ,1)/close
HIGH2 Ref(high ,2)/close
. . . . . .
Low/Close LOW1 Ref(low ,1)/close
LOW2 Ref(low ,2)/close
. . . . . .
Vwap/Close VWAP1 Ref(vwap ,1)/close
VWAP2 Ref(vwap ,2)/close
. . . . . .
Volume/Volume VOLUME1 Ref(volume ,1)/(volume + 1 e−12)
VOLUME2 Ref(volume ,2)/(volume + 1 e−12)
. . . . . .
2. Conduct a linear regression where the dependent variable is the factor sequence, and the
independent variables include the industry dummy variables and the market value variable.
3. Use the residual series from the regression as the neutralized factor sequence.
5.4. Factor Effectiveness Testing
The Information Coefficient (IC) is a statistical metric used to assess the correlation between
a factor and asset returns. In this study, the Spearman rank correlation coefficient is utilized to
calculate the IC value, referred to as RankIC.
For instance, consider the subdataset from the period of May 9, 2023, to August 2, 2023, for
IC value analysis. For each trading day within this period, IC values can be calculated for the
20

## Page 21

360 price-volume factors. The following discussion presents an example of IC values for 12 specific
price-volume factors as shown in Table 4.
Table 4: Factor IC Value Analysis
Factor Name Mean IC IC Standard Deviation Proportion of IC >0 (%) Proportion of |IC|>0.02 (%)
CLOSE1 0.098 0.038 99.61 98.82
CLOSE30 0.114 0.080 89.41 91.76
OPEN1 0.135 0.053 98.82 97.25
OPEN30 0.113 0.079 90.20 92.16
HIGH1 0.153 0.079 97.25 96.47
HIGH30 0.118 0.076 92.55 92.55
LOW1 0.089 0.075 87.84 89.80
LOW30 0.120 0.085 87.45 89.41
VWAP1 0.121 0.042 99.61 99.22
VWAP30 0.080 0.902 90.20 91.76
VOLUME1 0.102 0.060 93.33 90.98
VOLUME30 0.079 0.093 78.82 92.94
In multifactor stock selection models, an Information Coefficient (IC) value greater than 0.02 is
generally considered indicative that the factor partially reflects future stock trends, hence deemed
effective. As demonstrated in Table 4, the listed twelve factors all have mean IC values greater
than 0.07, with the proportion of IC absolute values exceeding 0.02 being over 89%, affirming their
effectiveness. Additionally, these factors exhibit over 85% positivity, indicating their predictions
align with market trends most of the time.
For other price-volume factors in this subdataset, and factors in other subdatasets, IC values
are analyzed in the same manner. If a factor’s mean IC is below 0.02, it is considered ineffective
and excluded from further analysis.
6. Experiments
This paper conducts a series of experiments to explore the performance of the Stockformer
model in predicting stock market returns and trend directions. Detailed analyses are carried out
addressing the following five research questions:
21
