# paper.pdf

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

## Page 22

6.1. Research Questions
1. Does the Stockformer model outperform existing baseline models in predicting returns and
market trends?
2. What impact do different components of the Stockformer model (such as the time attention
mechanism, graph embedding techniques, and multi-supervision learning) have on the model’s
performance?
3. How do hyperparameters affect the performance of Stockformer?
4. During the backtesting phase, considering the probability predictions of trend directions and
return forecasts may lead to different backtesting performances, which of these two outputs
should be chosen to maximize potential investment returns when labels are unknown (unable
to determine evaluation metrics)?
5. In the investment strategy backtesting phase, does the Stockformer model outperform existing
baseline models?
6.2. Experimental Setup
6.2.1. Dataset
As previously described in subsection 5.1, this section utilizes stocks from the CSI 300 Index,
which includes the top 300 stocks in terms of market liquidity and capitalization, reflecting the
overall trend of the A-share market. For the dataset, we use a matrix format where the rows
represent dates and the columns represent the return series of each of the 300 stocks. This serves
as the input for wavelet transformations. Based on the sign of the return rate at opening, bi-
nary up/down trend indicators (0 or 1) are generated. Alongside, 360 price-volume factors are
constructed according to the methods outlined in section 5, forming the multifactor input for our
model. Thus, the input dimensionality for our model is RT1×N×362.
The model uses data from the first 20 time steps ( T1= 20) to predict the returns and stock
price trends for the next two time steps ( T2= 2). These datasets are divided into training
(75%), validation (12.5%), and testing sets (12.5%), in chronological order and according to the
aforementioned day counts.
6.2.2. Evaluation Metrics
The evaluation metrics used in this study are divided into two main parts to comprehensively
assess the performance of the Stockformer model in stock market forecasting. The first part evalu-
22

## Page 23

ates the model’s predictive performance, focusing on its accuracy and efficiency in forecasting stock
returns and price trends. The second part evaluates the investment portfolio during the backtest-
ing phase of the investment strategy, involving the use of historical data to test the effectiveness of
model predictions in actual investment operations, assessing its performance and risk management
capabilities. This comprehensive evaluation aims to showcase the superiority of the Stockformer
model in stock return prediction and investment strategy formulation.
Predictive Performance Evaluation Metrics .To comprehensively assess model performance
across different prediction tasks, this study divides the evaluation metrics into two categories:
return prediction metrics and trend prediction metrics.
Return Prediction Metrics The following metrics are used to evaluate model performance
in the return prediction phase:
1.IC (Information Coefficient) (Lin, Zhou, Liu & Bian, 2021): The IC is a statistical mea-
sure that assesses the correlation between a single predictor and actual asset returns. It is
calculated using the Spearman correlation coefficient, reflecting the correlation between the
rankings of factors and returns. The formula is as follows:
IC =1
N(ˆy−mean( ˆy))T(y−mean( y))
std(ˆy)·std(y)(15)
2.ICIR (Information Coefficient Information Ratio) (Lin et al., 2021): ICIR measures
the stability and consistency of IC values, akin to the Sharpe ratio of IC, and is used to
evaluate the reliability of factor performance. The formula is as follows:
ICIR =mean(IC)
std(IC)(16)
3.RankIC (Li, Yang, Zhao, Bian, Qin & Liu, 2019): RankIC is an information coefficient based
on ranking, using the Spearman rank correlation coefficient instead of raw values, which is
more resistant to outliers. The formula is as follows:
RankIC =1
N(rank( ˆy)−mean(rank( ˆy)))T(rank( y)−mean(rank( y)))
std(rank( ˆy))·std(rank( y))(17)
4.RankICIR (Li et al., 2019): Information ratio of RankIC, calculated similarly to ICIR but
applied to RankIC values, used to assess the stability and predictive power of the rank-based
23

## Page 24

information coefficient. The formula is as follows:
RankICIR =mean(RankIC)
std(RankIC)(18)
Trend Prediction Metrics Used to evaluate model accuracy in predicting stock price
trends:
1.Directional Accuracy : This metric assesses the model’s accuracy in predicting the direc-
tion of stock price movements. For the Stockformer model, directional accuracy is calculated
using the trend classification output; for baseline models, it is determined by the sign (positive
or negative) of the predicted returns. The formula is as follows:
Directional Accuracy =Number of Correct Predictions
Total Number of Predictions×100% (19)
Portfolio Performance Metrics .This section introduces four key metrics used to assess the
performance of investment portfolios, aiding investors in better understanding the risk and return
characteristics of their investments.
1.Annualized Return : The annualized return helps investors comprehend the long-term
performance of an investment portfolio, reflecting the time value of the investment. The
formula is as follows:
Ry= (1 + R)250/(T2−T1)(20)
where Ris the total return, T1is the initial investment date, and T2is the closing date.
2.Maximum Drawdown : Maximum drawdown measures the largest loss in a portfolio during
a specific period and is used to assess the level of risk under market fluctuations. The formula
is as follows:
Max Drawdown =P−Q
P(21)
where Pis the highest net value during the period, and Qis the lowest net value following
the peak.
3.Annualized Volatility : Annualized volatility measures the variability of asset prices or an
investment portfolio over a given time period. The formula is as follows:
Volatility =√
252×σR (22)
where σRis the standard deviation of daily returns.
24

## Page 25

4.Sharpe Ratio (Sharpe, 1966): The Sharpe Ratio evaluates the return of an investment
portfolio relative to the risk it has taken on. The formula is as follows:
Sp=rp−rf
σP(23)
where rpis the return of the portfolio, rfis the risk-free rate (such as the yield on one-year
government bonds), and σPis the standard deviation of the portfolio’s returns.
6.2.3. Baselines
In this study, our proposed model, Stockformer, is compared with the following benchmark
models, which were introduced in the related work section. In total, ten baseline models are
utilized for comparison:
1. XGBoost(Chen & Guestrin, 2016): An optimized decision tree model using gradient boosting
techniques, widely used in classification and regression tasks.
2. LightGBM(Ke, Meng, Finley, Wang, Chen, Ma, Ye & Liu, 2017): A framework based on
gradient boosting that is characterized by fast speed, high efficiency, and friendliness to
large-scale data.
3. CatBoost(Dorogush, Ershov & Gulin, 2018): A high-performance gradient boosting tree
model that automatically handles categorical features, reducing the need for pre-processing
before model training.
4. LSTM(Hochreiter & Schmidhuber, 1997): A recurrent neural network capable of learning
long-term dependencies, widely applied in time series forecasting and natural language pro-
cessing.
5. GRU(Cho, Van Merri¨ enboer, Gulcehre, Bahdanau, Bougares, Schwenk & Bengio, 2014): A
simplified version of LSTM, offering similar performance but with fewer parameters and faster
training.
6. ALSTM(Wang & Hao, 2020): Enhances LSTM by incorporating an attention mechanism,
which improves the recognition of key information.
7. TCN(Bai, Kolter & Koltun, 2018): A network based on causal convolutions, effective in
handling time series data, suitable for scenarios requiring extensive historical information.
25

## Page 26

8. GATs(Velickovic, Cucurull, Casanova, Romero, Lio, Bengio et al., 2017): Graph Attention
Networks that allow nodes to dynamically compute their importance based on the features
of their neighbors through a graph attention mechanism.
9. Transformer(Vaswani et al., 2017): Based on self-attention mechanisms, capable of capturing
global dependencies, effectively processing sequence data.
10. Localformer(Zheng, Zhou, Ye & Zhan, 2023): By utilizing a Hilbert curve to unfold the image
matrix, it better preserves the smoothness of local information, suitable for images and other
two-dimensional data.
6.2.4. Parameter Settings
The Stockformer model was implemented using the PyTorch framework and trained with the
Adam optimizer for a total of 100 epochs. Due to computational resource constraints, the batch
size for input data was set to 2. Within the Stockformer model, the number of heads ein the
attention mechanism was set to 1, and the base dimension dewas set to 128. The number of layers
Lin the spatiotemporal encoder was set to 2. To capture high-frequency temporal dependencies,
dilated causal convolution layers with a kernel size of J= 2 were stacked. The initial learning
rate was set at 0.001, with a decay rate of 0.1 for adjusting the learning rate. Dropout was also
employed, with a dropout rate of 0.2, to mitigate the risk of overfitting.
6.2.5. Computational Environment
The computational setup for this study is detailed below:
1.CPU: Utilized two Intel(R) Xeon(R) Platinum 8352V CPUs @ 2.10GHz, each with 32 cores
and 64 threads, totaling 64 cores and 128 threads, providing substantial parallel computing
power. CPU clock speeds ranged from 800 MHz to 3500 MHz, with support for Intel VT-x
virtualization technology.
2.GPU: The system was equipped with an NVIDIA GeForce RTX 4090, boasting 24564 MB
of video memory.
3.Memory: The system included high-speed cache, comprising 3 MiB of L1d cache, 2 MiB of
L1i cache, 80 MiB of L2 cache, and 108 MiB of L3 cache, ensuring efficient data processing.
4.Operating System and Framework: All model training was conducted on a Linux system
based on the X86 64 architecture, utilizing the PyTorch deep learning framework.
26

## Page 27

This configuration provides an efficient and stable training environment for deep learning mod-
els.
6.3. Comparative Analysis of Predictive Performance (RQ 1)
This section will detail the prediction results of the Stockformer model on the test sets of 14
sub-datasets for stock return sequences and trend movements. We will compare the Stockformer
model with several advanced stock return prediction models identified in the research. These
baseline models are categorized into the following groups: the first category includes ensemble
learning models such as XGBoost, LightGBM, and CatBoost; the second category comprises re-
current neural network models including LSTM, GRU, and ALSTM; the third category consists
of component models of our system, namely Temporal Convolutional Networks (TCN) and Graph
Attention Networks (GATs); the fourth category encompasses models with an Encoder-Decoder
architecture that incorporate self-attention mechanisms, such as Transformer and Localformer.
Through this categorized comparison, we aim to thoroughly explore the performance advantages
of the Stockformer model across various evaluation metrics. The research findings are presented
in Table 5, which includes average values of various prediction performance metrics used across
the 14 sub-datasets, thereby visually demonstrating the comparative performance of Stockformer
against other state-of-the-art (SOTA) models.
The table 5 clearly demonstrates that Stockformer excels across multiple key performance indi-
cators. Notably, it significantly outperforms other models on the IC and ICIR metrics, indicating
a strong positive correlation and high stability and consistency between its predictive factors and
actual asset returns. Additionally, it achieves the highest Directional Accuracy at 57.46%, substan-
tially surpassing other models, further validating its accuracy and reliability in predicting market
trends.
When comparing the groups of baseline models, we observe the following characteristics:
1. The ensemble learning models (XGBoost, LightGBM, and CatBoost) have a speed advantage
when processing large datasets, but generally fall short in capturing complex market dynam-
ics compared to deep learning models. Although these models perform reasonably well in
Directional Accuracy, their poor performance on IC and ICIR metrics indicates limitations
in prediction accuracy and stability.
27

## Page 28

Table 5: Model Predictive Performance Comparison: Bold indicates the best performance, underlined indicates
the second best. An upward arrow ( ↑) signifies that higher metric values indicate better model performance, while
a downward arrow ( ↓) indicates that lower metric values are better. The table lists the average values of various
prediction performance metrics for the test set (out-of-sample predictions) on the 14 subdatasets.
Model IC ↑ Rank IC ↑ICIR↑Rank ICIR ↑Directional Accuracy ↑(%)
XGBoost -0.0077 -0.0104 -0.0594 -0.0788 51.11
LightGBM -0.0070 -0.0202 -0.0531 -0.1572 54.11
CatBoost -0.0214 -0.0269 -0.1609 -0.2135 52.25
LSTM -0.0014 -0.0038 -0.0168 -0.0372 53.04
GRU 0.0068 0.0076 0.0637 0.0674 52.17
ALSTM 0.0124 0.0081 0.0952 0.0603 52.85
TCN 0.0019 -0.0004 0.0190 0.0015 53.83
GATs 0.0108 0.0059 0.0839 0.0475 54.52
Transformer 0.0061 0.0048 0.0511 0.0418 49.29
Localformer 0.0071 0.0104 0.0646 0.0928 54.87
Stockformer 0.0294 0.0344 0.1921 0.2669 57.46
2. Recurrent neural network models (LSTM, GRU, and ALSTM) exhibit strong performance
in capturing the temporal dependencies in time series data. Among these, ALSTM stands
out in all baseline models, particularly on the IC and ICIR metrics, demonstrating high
predictive consistency and stability. The inclusion of an attention mechanism substantially
contributes to ALSTM’s impressive performance, enhancing its ability to effectively process
and parse critical information in time series data, especially in complex and long-term depen-
dencies. The addition of the attention layer not only helps the model focus on the historical
information most critical to future predictions but also improves the model’s adaptability to
dynamic changes in the data. Furthermore, ALSTM’s performance reaffirms the correctness
of our multiple uses of attention mechanisms in the design of Stockformer.
3. Models utilizing Encoder-Decoder architectures with integrated self-attention mechanisms,
such as Transformer and Localformer, are theoretically expected to excel in capturing long-
distance dependencies. In terms of performance metrics for rate of return predictions, these
28

## Page 29

models demonstrate some stability, yet they do not exhibit superior advantages. Notably, in
the performance metrics for trend prediction accuracy, Transformer performs poorly, revealing
potential limitations in its ability to predict actual financial market trends. This suggests
that when employing these models, a greater emphasis may be required on model tuning and
feature adaptation to specific scenarios.
Advantages of Stockformer .The exceptional performance of Stockformer can be attributed
to its unique architecture that integrates various powerful deep learning technologies, thereby
optimizing the model’s ability to process complex stock market data. Specifically:
1.Temporal Convolutional Networks (TCN) provide effective capture of long-term de-
pendencies in time series data.
2.Graph Attention Mechanisms (GATs) enable the model to consider interactions and
mutual influences among stocks, enhancing its understanding of market dynamics.
3.Self-Attention Mechanisms (Transformer and Localformer) strengthen the model’s
ability to capture complex interactions among various points in time series data.
4.Multi-Supervision Learning : Through a sophisticated multi-task learning framework,
Stockformer can simultaneously perform classification and regression tasks, allowing classifi-
cation and regression to supervise each other and perform multi-dimensional predictions on
stock trends and returns, enhancing the model’s adaptability and predictive accuracy.
In subsequent model backtesting and comparisons, this study selected the best-performing
models from each baseline category — LightGBM, ALSTM, TCN, GATs, Localformer, and our
proprietary Stockformer model — for detailed analysis.
6.4. Ablation Study (RQ 2)
To investigate the effects of different components within the Stockformer model, this study
compares it against six distinct variants:
1. ”w/o D (Decoupling Flow Layer)”: Stockformer without the Decoupling Flow Layer;
2. ”w/o T (Dilated causal convolution and Temporal Attention)”: Stockformer lacking both
Dilated Causal Convolution and Temporal Attention;
3. ”w/o G (Graph)”: Stockformer devoid of spatial and temporal graph embeddings;
29

## Page 30

4. ”w/o F (Fusion)”: Stockformer where Fusion Attention is replaced by addition operations;
5. ”w/o Reg”: Stockformer that outputs only the classification task, using only the classification
results as the supervisory signal;
6. ”w/o Cla”: Stockformer that outputs only the regression task, using only the regression
results as the supervisory signal.
Table 6: Stockformer Ablation Study: Bold indicates the best performance, underlined indicates the second-best
performance. An upward arrow ( ↑) indicates that higher metric values denote better model performance, while a
downward arrow ( ↓) indicates that lower metric values are preferable. The table lists the average values of various
prediction performance metrics for the test set (out-of-sample predictions) on the 14 subdatasets.
Variant IC( ↑) Rank IC( ↑) ICIR( ↑) Rank ICIR( ↑) Directional Accuracy( ↑)(%)
w/o D 0.0122 0.0202 0.0504 0.0663 52.81
w/o T 0.0122 0.0205 0.0506 0.0840 52.65
w/o G 0.0104 0.0197 0.0498 0.0742 51.83
w/o F 0.0186 0.0267 0.0641 0.0928 50.30
w/o Reg 0.0205 0.0281 0.0888 0.1005 54.60
w/o Cla 0.0237 0.0291 0.1233 0.1215 53.32
Stockformer 0.0294 0.0344 0.1921 0.2669 57.46
The data in Table 6 clearly demonstrate that the complete Stockformer model outperforms
its various variants, highlighting the importance of each component within the model. Removing
or modifying key components such as the Decoupling Flow Layer, Dilated Causal Convolution,
Graph Embeddings, or Fusion Attention leads to a noticeable decline in performance. Particularly,
the removal of Graph Embeddings significantly impacts the model’s accuracy and error metrics,
underscoring the critical role of modeling complex data relationships for prediction accuracy. Fur-
thermore, Stockformer’s strategy of Multi-Supervision, which involves handling both classification
and regression tasks concurrently, enhances the model’s generalization capabilities and adaptabil-
ity. This multi-task learning approach not only provides the model with a richer learning signal
but also aids in capturing and understanding the complexities of data across multiple dimensions,
thereby improving overall predictive performance.
30

## Page 31

6.5. Hyperparameter Sensitivity Analysis (RQ 3)
3264 128 256
Hidden Layer Size2.02.22.42.62.8IC1e2
IC
1 2 3 4
Layers Number1.82.02.22.42.62.83.0IC1e2
IC
816 32 64
Batch Size2.42.62.8IC1e2
IC
1.0 1.5 2.0 2.5
Lambda2.22.42.62.8IC1e2
IC0.050.100.150.20
ICIR
ICIR
0.140.150.160.170.180.19
ICIR
ICIR
0.1700.1750.1800.1850.190
ICIR
ICIR
0.1600.1650.1700.1750.1800.1850.190
ICIR
ICIR
Figure 4: Hyperparameter Sensitivity Analysis. In the figure, the blue line represents the Information Coefficient
(IC) values on the test set (out-of-sample predictions), while the red line represents the Information Coefficient
Information Ratio (ICIR) values. The metrics evaluate the impact of various hyperparameters including hidden
layer size, number of encoder layers, batch size, and classification loss weight on the model’s prediction accuracy and
stability.
In this study, we conducted a sensitivity analysis of hyperparameters for the Stockformer model,
covering hidden layer sizes (32, 64, 128, 256), number of encoder layers (1 to 4), batch sizes (8, 16,
32, 64), and classification loss weights (1.0, 1.5, 2.0, 2.5), as shown in Figure 4. The various settings
of these hyperparameters were thoroughly examined for their impact on key performance indicators
such as Information Coefficient (IC) and Information Coefficient Information Ratio (ICIR), with the
31

## Page 32

aim of identifying optimal configurations to enhance the model’s prediction accuracy and stability.
The analysis results indicate that optimal performance of the Stockformer is achieved when
the hidden layer size is set to 128 and the batch size to 16. These findings highlight the ideal
combination of model parameters under specific settings, and how further increasing or decreasing
these values may lead to performance degradation. The best performance is observed when the
number of encoder layers is set to 2, suggesting that a moderate increase in model depth can
effectively enhance performance, but additional layers may not yield further benefits. In adjusting
the weight for the classification task, a value of 2.0 shows the best performance, emphasizing the
importance of appropriately adjusting loss weights in balancing classification and regression tasks.
6.6. Backtesting Input Selection (Research Question 4)
In the backtesting phase of stock market prediction models (forecasting future outcomes),
selecting the appropriate output of Stockformer—either classification or regression results—for
backtesting is crucial. This study compares the backtesting performance of different outputs—the
probability of trend predictions (classification output) and the results of return predictions (re-
gression output)—to determine which output can maximize potential investment returns. The
following analysis and recommendations are provided:
High Confidence Prediction Proportion Analysis .The proportion of high confidence pre-
dictions is a metric used to measure the certainty of model predictions, defined in this study as the
proportion of upward trend predictions (label 1) with a probability value greater than 0.6 or less
than 0.4. This proportion reflects the level of certainty in the model’s predictions. A higher propor-
tion indicates a higher certainty in the model’s classification predictions, while a lower proportion
indicates greater uncertainty. Based on the backtesting results from 14 sub-datasets created by
sliding window segmentation (detailed backtesting results are plotted in the next subsection), we
obtain Table 7, comparing the performance of regression and classification outputs across each
sub-dataset.
Model Selection Recommendations .Based on the analysis of high confidence prediction pro-
portions and optimal outcomes over different time ranges, we offer the following recommendations:
1.When high confidence prediction proportion is high : Classification outputs generally
provide better results. For instance, when the proportion of high confidence predictions
32

## Page 33

Table 7: High Confidence Prediction Proportion and Best Outcome Analysis
Backtesting Date Range High Confidence Prediction Proportion Best Result
2018-03-01 – 2020-10-29 0.6886 cla
2018-05-31 – 2021-01-25 0.6797 cla
2018-08-27 – 2021-04-28 0.2558 cla
2018-11-28 – 2021-07-28 0.1139 reg
2019-03-04 – 2021-11-01 0.0388 reg
2019-06-03 – 2022-01-26 0.2422 cla
2019-08-28 – 2022-05-05 0.6407 cla
2019-11-29 – 2022-08-01 0.3771 cla
2020-03-04 – 2022-11-02 0.3183 cla
2020-06-03 – 2023-02-03 0.7132 cla
2020-08-31 – 2023-05-05 0.0292 reg
2020-12-02 – 2023-08-02 0.2599 cla
2021-03-05 – 2023-11-03 0.1180 reg
2021-06-04 – 2024-01-30 0.1297 reg
exceeds 20%, classification outcomes often demonstrate higher prediction accuracy. This
may be attributed to the classification model’s effectiveness in capturing market trends under
these conditions.
2.When high confidence prediction proportion is low : Regression outputs may be a
more suitable choice. In scenarios where prediction confidence is low, regression models can
better handle the uncertainty and volatility in the data, providing more continuous and stable
outputs.
These strategies can assist investors and analysts in maximizing investment returns during
the backtesting phase when using Stockformer’s results by selecting the most appropriate model
output.
6.7. Strategy Backtesting
6.7.1. Topk-Dropout Strategy
The Topk-Dropout strategy is a ranking-based quantitative stock selection strategy that con-
structs a portfolio by ranking stocks based on their factor values and retaining only the top k stocks
to achieve stock selection. Specifically, the Topk-Dropout strategy includes the following steps:
33

## Page 34

High Score
Low ScoreCurrentTradingAfter TradingDrop = 3, TopK = 5The stock that is heldThe stock that will continue to be heldThe stock that will be boughtThe stock that will  be soldOther stockFigure 5: In the TopKDropout strategy, with TopK as 5 and Drop as 3: Prior to the subsequent position adjustment,
the three lowest-ranked stocks are discarded. From the remaining stocks, excluding the current holdings, the top
three with the highest scores (given by TopK-Drop, i.e., 5-3) are chosen for the next trading cycle.
1. On the current trading day, calculate the factor values for each stock given a set of factors.
2. Sort the stocks by factor values, sell stocks that are ranked beyond the top k in prediction
scores, and buy an equal number of stocks that are newly ranked in the top k, ensuring that
exactly k stocks are held at all times, except at the start of trading when this number is zero.
3. Allocate funds to the selected stocks equally by weight and conduct buying and selling ac-
cording to the trading frequency.
4. At the next rebalancing period, re-sort the stocks by factor values and select stocks, repeating
the steps above.
The core idea of the Topk-Dropout strategy is to select stocks that rank high in factor values,
assuming these stocks are more likely to perform well in the future. Unlike other stock-picking
strategies, the TopK-Dropout strategy focuses on investing in only the top k ranked stocks and
discards the rest. This method enhances the concentration of the investment portfolio, reduces
transaction costs, and also decreases the number of low-quality stocks in the portfolio, thus im-
proving the effectiveness and stability of the stock-picking strategy.
Figure 5 shows an example of the TopK-Dropout strategy where k is set to 5. Before the next
position adjustment, the three lowest-ranked stocks are discarded. From the remaining stocks,
34

## Page 35

excluding the current holdings, the top k stocks with the highest scores are chosen for the next
trading cycle. The investment strategy settings adopted in this paper are as follows:
1. The period from 2018 to 2024 is divided into 14 time intervals, with independent backtesting
conducted on each interval.
2. The trading strategy chosen is TopK-Dropout, with k set to 5.
3. The stock pool is selected from the screened constituents of the CSI 300 index.
4. The benchmark for comparison is the CSI 300 index.
5. The trading frequency is daily, with the model predicting the T+1 day return rate, i.e., the
price change from the T+1 day closing price to the T+2 day closing price.
6. The transaction cost is 0.1% per side, with stamp duty at 0.1% before August 27, 2023, and
0.05% thereafter.
6.7.2. Investment Strategy Backtesting
/uni00000012/uni00000010/uni00000011/uni00000017/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000011/uni00000018/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000011/uni00000018/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000011/uni00000018/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000011/uni00000019/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000011/uni00000019/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000011/uni00000019/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000014/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010
/uni00000024/uni00000041/uni00000054/uni00000045/uni00000013/uni00000010/uni00000010/uni00000010/uni00000013/uni00000015/uni00000010/uni00000010/uni00000014/uni00000010/uni00000010/uni00000010/uni00000014/uni00000015/uni00000010/uni00000010/uni00000015/uni00000010/uni00000010/uni00000010/uni00000015/uni00000015/uni00000010/uni00000010/uni00000023/uni0000004c/uni0000004f/uni00000053/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000030/uni00000052/uni00000049/uni00000043/uni00000045/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010/uni00000060/uni00000029/uni0000004e/uni00000044/uni00000045/uni00000058/uni00000060/uni00000023/uni0000004c/uni0000004f/uni00000053/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000030/uni00000052/uni00000049/uni00000043/uni00000045/uni00000060/uni00000034/uni00000052/uni00000045/uni0000004e/uni00000044
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010/uni00000060/uni00000023/uni0000004c/uni0000004f/uni00000053/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000030/uni00000052/uni00000049/uni00000043/uni00000045
Figure 6: The Trend of the CSI 300 Index from 2018 to 2024
Figure 6 shows the trend of the CSI 300 Index from 2018 to 2024, which largely reflects the
overall market trend. During this period, the market exhibited phases of rise, fall, and fluctuation.
35

## Page 36

The stability of investment strategies under different market conditions is a critical issue that needs
attention.
Table 8: Selection of Backtesting Intervals
Backtesting Start Date Backtesting End Date Market Condition
2020-11-02 2021-01-25 Uptrend
2022-01-28 2022-05-05 Downtrend
2023-05-09 2023-08-02 Fluctuation
This study selects three representative periods of uptrend, downtrend, and fluctuation for
backtesting analysis from the 14 divided intervals, as shown in Table 8.
/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000011/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000012/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000010/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000011/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000012/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000012/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000010/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000011/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000015
/uni00000024/uni00000041/uni00000054/uni00000045/uni00000011/uni0000000e/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000011/uni0000000e/uni00000012/uni00000011/uni0000000e/uni00000013/uni00000011/uni0000000e/uni00000014/uni00000011/uni0000000e/uni00000015/uni00000011/uni0000000e/uni00000016/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000022/uni00000041/uni00000043/uni0000004b/uni00000054/uni00000045/uni00000053/uni00000054/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000029/uni0000004e/uni00000054/uni00000045/uni00000052/uni00000056/uni00000041/uni0000004c/uni0000001a/uni00000060/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000012/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000015
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
Figure 7: Out-of-sample investment portfolio net value curves during the uptrend period (2020-11-02 to 2021-01-
25). Bold lines represent the CSI 300 index benchmark (black), regression task output net value series (red),
and classification task output net value series (blue). Thin lines represent selected outstanding baseline models
(LightGBM, ALSTM, TCN, GATs, Localformer).
Model Backtesting Performance During Uptrend .During the backtesting of the uptrend
market, this study selected the period from November 2, 2020, to January 25, 2021. The models
used included Stockformer’s regression output (Stockformer reg) and classification output (Stock-
former cla), along with other selected high-performing baseline models. To determine which output
type should be used during the backtesting period, we referred to the analysis of the high confi-
36

## Page 37

dence prediction proportion indicator discussed in subsection 6.6. The results indicate that using
Stockformer’s classification output (Stockformer cla) for predictions was more appropriate during
this period. Furthermore, the CSI 300 index was used as a benchmark, as shown in Figure 7. All
models’ backtested net values were generally higher than the benchmark index. Particularly, the
Stockformer model consistently outperformed other models in net value during most of the period,
highlighting its superior performance and stability in uptrend markets.
Table 9: Investment Portfolio Performance during the Uptrend: Bold indicates best performance, underlined indi-
cates second best. An upward arrow ( ↑) indicates better performance with higher values, and a downward arrow ( ↓)
indicates better performance with lower values. This table lists various performance metrics for investment portfolios
during the uptrend period in the test dataset (out-of-sample prediction) from November 2, 2020, to January 25, 2021.
Benchmark &
ModelAnnualized
Return (%) ↑Annualized
Volatility (%) ↓Maximum
Drawdown (%) ↓Sharpe Ratio
(%)↑
CSI 300 80.52 16.46 3.5 4.73
LightGBM 270.45 29.35 4.41 9.12
ALSTM 229.56 26.82 3.63 7.96
TCN 234.5 27.43 4.98 8.45
GATs 118.7 25.48 7.16 4.55
Localformer 217.31 26.91 3.79 7.98
Stockformer cla 239.73 29.78 3.07 8.46
During the uptrend market period from November 2, 2020, to January 25, 2021, various pre-
diction models demonstrated distinct performances as shown in Table 9. LightGBM led with the
highest annualized return of 270.45%, indicating significant profitability during this period. AL-
STM and TCN, with annualized returns of 229.56% and 234.5% respectively, also exhibited high
profitability but were accompanied by relatively high volatility, which may increase investment
risk. GATs showed good risk control capabilities with the lowest annual volatility of 25.48%, al-
though its returns were comparatively lower. Localformer achieved a better balance between risk
and returns.
In contrast, the Stockformer model also performed impressively during this uptrend, achiev-
ing an annualized return of 239.73%, second only to LightGBM. More importantly, Stockformer’s
maximum drawdown was only 3.07%, significantly lower than other models, demonstrating excel-
37

## Page 38

lent resilience during market downturns. Additionally, its Sharpe ratio of 8.46% ranked second
among all models, proving its efficiency in risk-adjusted returns. Overall, Stockformer not only
excelled in profitability but also showcased significant advantages in risk management and market
adaptability, making it an ideal choice in uptrend conditions.
/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000011/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000012/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000010/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000011/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000012/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000012/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000010/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000011/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000011/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000012/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000013
/uni00000024/uni00000041/uni00000054/uni00000045/uni00000010/uni0000000e/uni00000017/uni00000015/uni00000010/uni0000000e/uni00000018/uni00000010/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000015/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000022/uni00000041/uni00000043/uni0000004b/uni00000054/uni00000045/uni00000053/uni00000054/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000029/uni0000004e/uni00000054/uni00000045/uni00000052/uni00000056/uni00000041/uni0000004c/uni0000001a/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000018/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000015
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
Figure 8: Out-of-sample investment portfolio net value curves during the downtrend period (2022-01-28 to 2022-
05-05). Bold lines represent the CSI 300 index benchmark (black), regression task output net value series (red),
and classification task output net value series (blue). Thin lines represent selected outstanding baseline models
(LightGBM, ALSTM, TCN, GATs, Localformer).
Model Backtesting Performance During Downtrend .During the analysis of backtesting
performance in downtrend conditions, this study focused on the period from January 28, 2022, to
May 5, 2022. For this interval, the Stockformer model’s classification task output (Stockformer cla)
and other distinguished baseline models were employed for comparison. All models exhibited a
declining trend in their net value curves, reflecting the overall negative market trend. However, as
shown in Figure 8, compared to other models and the CSI 300 index, Stockformer and TCN showed
smaller declines in net value, demonstrating their relative resilience and stable excess returns during
market downturns.
Table 10 presents the performance of various investment portfolios corresponding to each model
during the downturn market conditions within the backtesting period (January 28, 2022, to May
5, 2022). Multiple stock prediction models displayed varying performances. LightGBM and GATs,
despite showing lower decreases in annualized returns (-39.25% and -34.77%, respectively), faced
38

## Page 39

Table 10: Investment Portfolio Performance During the Downtrend: Bold indicates best performance, underlined
indicates second best. An upward arrow ( ↑) indicates better performance with higher values, and a downward arrow
(↓) indicates better performance with lower values. This table lists various performance metrics for investment
portfolios during the downtrend period in the test dataset (out-of-sample prediction) from January 28, 2022, to May
5, 2022.
Benchmark &
ModelAnnualized
Return (%) ↑Annualized
Volatility (%) ↓Maximum
Drawdown (%) ↓Sharpe Ratio
(%)↑
CSI 300 -50.94 26.67 18.66 -1.99
LightGBM -39.25 42.61 27.52 -0.97
ALSTM -59.85 43.62 26.63 -1.42
TCN -6.72 28.66 14.43 -0.31
GATs -34.77 38.11 23.15 -0.97
Localformer -64.46 43.41 29.25 -1.53
Stockformer cla -15.18 31.04 19.60 -0.56
higher annualized volatility and maximum drawdowns, indicating significant pressure under market
fluctuations. ALSTM and Localformer exhibited even steeper declines and higher volatility, further
highlighting the challenges in extreme market conditions.
In contrast, the Stockformer model demonstrated noticeable stability during this market down-
turn. It experienced a lesser decline in annualized returns (-15.18%), ranking second best among all
models, which illustrates its robust ability to resist market downturns. Additionally, Stockformer’s
maximum drawdown was 19.60%, which, although not the lowest, still performed relatively well
compared to other models. Regarding volatility, Stockformer also reported the second lowest an-
nualized volatility (31.04%), further confirming its robustness relative to other models.
Model Backtesting Performance During Sideways Market .During the analysis of market
performance in a sideways market condition, this study focused on the period from May 9, 2023,
to August 2, 2023. For this interval, we utilized the classification output of the Stockformer
model (Stockformer cla) along with other exemplary baseline models for comparison. As shown in
Figure 9, during the sideways market conditions, Stockformer demonstrated a significant advantage.
Its backtesting net value was positive, unlike most other models which predominantly showed
negative net values. Furthermore, compared to the CSI 300 index, Stockformer was able to provide
39

## Page 40

/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000011/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000013/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000010/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000011/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000012/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000012/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000010/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000011/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000011/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000012/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000012
/uni00000024/uni00000041/uni00000054/uni00000045/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000022/uni00000041/uni00000043/uni0000004b/uni00000054/uni00000045/uni00000053/uni00000054/uni00000049/uni0000004e/uni00000047/uni00000060/uni00000029/uni0000004e/uni00000054/uni00000045/uni00000052/uni00000056/uni00000041/uni0000004c/uni0000001a/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000019/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000012
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033Figure 9: Out-of-sample investment portfolio net value curves during the sideways market (2023-05-09 to 2023-
08-02). Bold lines represent the CSI 300 index benchmark (black), regression task output net value series (red),
and classification task output net value series (blue). Thin lines represent selected outstanding baseline models
(LightGBM, ALSTM, TCN, GATs, Localformer).
more stable excess returns, showcasing its exceptional adaptability and robustness in handling
market fluctuations.
The performance of investment portfolios during sideways market conditions is presented in Ta-
ble 11. During this phase, most models failed to achieve positive returns, reflecting the challenging
nature of sideways markets on model stability. LightGBM, ALSTM, TCN, GATs, and Localformer
all exhibited high annualized volatility and maximum drawdowns, with ALSTM and Localformer
performing the worst, experiencing annualized returns dropping to -22.51% and -38.25% respec-
tively. This indicates extreme sensitivity in sideways markets, leading to significant losses.
In contrast, Stockformer significantly outperformed other models under these market condi-
tions, with an annualized return of 44.48%, and maintained lower annualized volatility (15.68%)
and the smallest maximum drawdown (4.14%). Its Sharpe ratio reached 2.71, far surpassing all
other models. This demonstrates that Stockformer not only managed to maintain positive returns
during market fluctuations but also effectively controlled risk and preserved high return stability.
This highlights Stockformer’s superior performance in managing market uncertainties and its high
risk-adjusted returns, making it an ideal choice for investing in sideways markets.
40

## Page 41

Table 11: Investment Portfolio Performance During Sideways Market Conditions: Bold indicates the best perfor-
mance, underlined indicates the second best. An upward arrow ( ↑) indicates better performance with higher values,
and a downward arrow ( ↓) indicates better performance with lower values. This table lists various performance met-
rics for investment portfolios during the sideways market conditions in the test dataset (out-of-sample prediction)
form May 9, 2023, to August 2, 2023.
Benchmark &
ModelAnnualized
Return (%) ↑Annualized
Volatility (%) ↓Maximum
Drawdown (%) ↓Sharpe Ratio
(%)↑
CSI 300 -6.05 14.30 5.92 -0.56
LightGBM -17.64 25.28 8.65 -0.78
ALSTM -22.51 30.94 13.52 -0.79
TCN -15.31 28.26 10.82 -0.61
GATs -21.57 23.09 9.10 -1.02
Localformer -38.25 32.86 18.61 -1.22
Stockformer cla 44.48 15.68 4.14 2.71
/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000013/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000011/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000012/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000011/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000012/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000011/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000012/uni00000018/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000015/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000013/uni00000010/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000012/uni00000019
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000010/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000013/uni00000011/uni0000000e/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000011/uni0000000e/uni00000012/uni00000011/uni0000000e/uni00000013/uni00000011/uni0000000e/uni00000014/uni00000011/uni0000000e/uni00000015/uni00000011/uni0000000e/uni00000016/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000010/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000012/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000015
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000012/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000011/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000012/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000012/uni00000014/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000017/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000012/uni00000018
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000011/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000011/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000013/uni00000011/uni00000010/uni0000000e/uni00000019/uni00000011/uni0000000e/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000011/uni0000000e/uni00000012/uni00000011/uni0000000e/uni00000013/uni00000011/uni0000000e/uni00000014/uni00000011/uni0000000e/uni00000015/uni00000011/uni0000000e/uni00000016/uni00000011/uni0000000e/uni00000017/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000013/uni00000010/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000012/uni00000018
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000012/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000011/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000012/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000012/uni00000014/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000015/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000013/uni00000010/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000011
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000011/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000012/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000011/uni00000010 /uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000012/uni00000015 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000014/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000011/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000013/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000016
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000010/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000011/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000013/uni00000010/uni0000000e/uni00000017/uni00000015/uni00000010/uni0000000e/uni00000018/uni00000010/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000015/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000018/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000015
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000011/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000012/uni00000011/uni0000000e/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000011/uni0000000e/uni00000012/uni00000011/uni0000000e/uni00000013/uni00000011/uni0000000e/uni00000014/uni00000011/uni0000000e/uni00000015/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000019/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000011
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000011/uni00000010/uni0000000e/uni00000017/uni00000015/uni00000010/uni0000000e/uni00000018/uni00000010/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000013/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000012
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000010/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000011/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000015/uni00000011/uni0000000e/uni00000012/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000012/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000014/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000013
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000012/uni00000012 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000010/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000013/uni0000006d/uni00000012/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000010/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000014/uni0000006d/uni00000012/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000018/uni00000010/uni0000000e/uni00000019/uni00000011/uni0000000e/uni00000010/uni00000011/uni0000000e/uni00000011/uni00000011/uni0000000e/uni00000012/uni00000011/uni0000000e/uni00000013/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000012/uni0000006d/uni00000010/uni00000017/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000015
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000011/uni00000017 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000016/uni0000006d/uni00000011/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000010/uni00000011 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000011/uni00000016 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000017/uni0000006d/uni00000013/uni00000011/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni00000011/uni0000000e/uni00000010/uni00000015/uni00000011/uni0000000e/uni00000011/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000015/uni0000006d/uni00000010/uni00000019/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000012
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000011/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000019/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000010/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000010/uni0000006d/uni00000011/uni00000018 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000012/uni00000010/uni0000000e/uni00000017/uni00000015/uni00000010/uni0000000e/uni00000018/uni00000010/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000010/uni00000018/uni0000006d/uni00000010/uni00000014/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000013
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000011/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000012/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000011/uni00000014 /uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000012/uni0000006d/uni00000012/uni00000019 /uni00000012/uni00000010/uni00000012/uni00000014/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000011/uni00000013 /uni00000012/uni00000010/uni00000012/uni00000014/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000012/uni00000018/uni00000010/uni0000000e/uni00000017/uni00000015/uni00000010/uni0000000e/uni00000018/uni00000010/uni00000010/uni0000000e/uni00000018/uni00000015/uni00000010/uni0000000e/uni00000019/uni00000010/uni00000010/uni0000000e/uni00000019/uni00000015/uni00000011/uni0000000e/uni00000010/uni00000010/uni0000002e/uni00000045/uni00000054/uni00000060/uni00000036/uni00000041/uni0000004c/uni00000055/uni00000045/uni00000012/uni00000010/uni00000012/uni00000013/uni0000006d/uni00000011/uni00000011/uni0000006d/uni00000010/uni00000017/uni00000060/uni0000006d/uni0000006d/uni00000060/uni00000012/uni00000010/uni00000012/uni00000014/uni0000006d/uni00000010/uni00000011/uni0000006d/uni00000013/uni00000010
/uni0000002d/uni0000004f/uni00000044/uni00000045/uni0000004c
/uni00000023/uni00000033/uni00000029/uni00000060/uni00000013/uni00000010/uni00000010
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000052/uni00000045/uni00000047
/uni00000033/uni00000054/uni0000004f/uni00000043/uni0000004b/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052/uni0000003f/uni00000043/uni0000004c/uni00000041
/uni0000002c/uni00000049/uni00000047/uni00000048/uni00000054/uni00000027/uni00000022/uni0000002d/uni00000034/uni00000023/uni0000002e
/uni0000002c/uni0000004f/uni00000043/uni00000041/uni0000004c/uni00000046/uni0000004f/uni00000052/uni0000004d/uni00000045/uni00000052
/uni00000021/uni0000002c/uni00000033/uni00000034/uni0000002d
/uni00000027/uni00000021/uni00000034/uni00000033
Figure 10: Comprehensive chart of net value changes for investment portfolios in various time intervals based on
out-of-sample predictions. Bold lines represent the CSI 300 index benchmark (black), Stockformer’s regression task
output net value series (red), and classification task output net value series (blue). Thin lines display the net value
series for selected outstanding baseline models including LightGBM, ALSTM, TCN, GATs, and Localformer.
41

## Page 42

Summary Analysis of Backtesting Strategy .From the backtesting analysis across uptrend,
downtrend, and sideways market conditions, it is evident that Stockformer demonstrates signifi-
cant advantages over other models. In uptrend markets, Stockformer showed strong profitability,
closely matching LightGBM, but exhibited superior performance in terms of risk control and max-
imum drawdown. In downtrend markets, although all models experienced declines in net value,
Stockformer’s losses were noticeably smaller than those of other models, indicating higher mar-
ket adaptability and stability. In sideways markets, Stockformer was the only model to achieve
positive returns, and its risk-adjusted returns (Sharpe ratio) far exceeded those of other models,
emphasizing its ability to maintain consistent profitability in complex market environments.
Table 12: Comparison of investment strategy performance over multiple periods with the CSI 300 Index: Analysis
of annualized return rates, Sharpe ratios, and maximum drawdowns
Backtesting Period Annualized Return (%) ↑Maximum Drawdown (%) ↓ Sharpe Ratio (%) ↑
Stockformer CSI 300 Stockformer CSI 300 Stockformer CSI 300
2020-08 – 2020-10 0.4703 0.1053 0.0772 0.0580 1.5611 0.4611
2020-11 – 2021-01 2.3973 0.8052 0.0307 0.0350 7.9580 4.7254
2021-02 – 2021-04 0.0107 -0.3106 0.0953 0.1518 -0.0611 -1.4306
2021-05 – 2021-07 1.2751 -0.2976 0.0438 0.1105 5.0450 -1.7625
2021-08 – 2021-10 0.6465 0.0694 0.1087 0.0543 2.4503 0.2964
2021-11 – 2022-01 -0.0588 -0.0948 0.0882 0.0797 -0.4133 -0.8723
2022-02 – 2022-05 -0.1518 -0.5094 0.1960 0.1866 -0.5566 -1.9882
2022-05 – 2022-08 1.1880 0.3371 0.1061 0.0725 4.0540 1.9663
2022-08 – 2022-11 -0.1703 -0.4019 0.1559 0.1680 -0.7093 -2.2679
2022-11 – 2023-02 0.8131 0.4175 0.0688 0.0425 3.7037 2.6248
2023-02 – 2023-05 0.1748 -0.0794 0.0783 0.0497 0.9286 -0.7705
2023-05 – 2023-08 0.4448 -0.0605 0.0414 0.0592 2.7118 -0.5607
2023-08 – 2023-11 -0.3799 -0.4559 0.1011 0.1359 -2.6577 -3.556
2023-11 – 2024-01 -0.4633 -0.4348 0.1268 0.1107 -2.5537 -3.2945
From July 2020 to January 2024, the investment strategy proposed in this paper was meticu-
lously backtested across 14 distinct time intervals in sub-datasets. The net value curves of these
models are presented in Figure 10, while Table 12 provides detailed backtesting performance across
these intervals. The results indicate that, except for the period from November 2023 to January
42

## Page 43

2024, the strategy’s annualized return rate outperformed the CSI 300 index in the remaining 13
intervals. Moreover, the Sharpe ratio exceeded that of the CSI 300 index in all 14 intervals, and the
maximum drawdown was similar to that of the CSI 300, demonstrating the strategy’s effective risk
control. From the various sub-figures, it is observable that the Stockformer model secured stable
excess returns in all test intervals and, in most cases, its portfolio net value performance surpassed
other selected outstanding baseline models such as LightGBM, ALSTM, etc. In summary, the
investment strategy described in this paper demonstrates exceptional stability and considerable
excess returns relative to the CSI 300 index across varying market conditions, whether in rising,
falling, or volatile markets.
These results demonstrate that the Stockformer model exhibits outstanding stability and relia-
bility across diverse market conditions. Particularly during market downturns or volatile periods,
it maintains high performance levels, showing a high degree of adaptability to market fluctuations.
This capability allows Stockformer to be suitable not only for strategies aiming for high returns but
also excels in environments requiring stringent risk control. Hence, whether the investment goal is
capital appreciation or capital preservation, Stockformer has proven to be a robust investment tool
capable of providing solid support under various market conditions. Furthermore, for investors
seeking consistent performance across different market environments, Stockformer offers an effec-
tive strategy option. Through detailed backtesting verification, Stockformer has demonstrated its
potential and practicality as an advanced investment strategy.
7. Conclusion and Future Work
7.1. Conclusion
This study introduces Stockformer, a stock selection model that integrates wavelet transform
and multitask self-attention networks, aiming to enhance the precision and adaptability of ana-
lyzing the complex dynamics of global securities markets. By extensively mining data from the
stocks within the CSI 300 index and partitioning it into 14 subdatasets, and by employing 360 rig-
orously tested price-volume factors, the model significantly improves the interpretation of market
fluctuations and abrupt events. Stockformer utilizes advanced deep learning technologies, includ-
ing dual-frequency spatiotemporal encoder, graph embedding techniques, and multitask learning
strategies. These technologies not only increase the accuracy of stock return and market trend
43

## Page 44

predictions but also enhance the model’s capability to detect market dynamics in detail. Specif-
ically, the model precisely captures short-term market fluctuations and long-term trends through
high-low frequency decomposition techniques, while the integrated dual-frequency spatiotemporal
encoder ensures effective processing of temporal and spatial dependencies. Experimental results
demonstrate that Stockformer outperforms ten baseline models across various predictive perfor-
mance metrics. Notably, it achieves a directional accuracy of 57.46%, significantly higher than
other models. Moreover, the study conducts backtesting on different stock selection models, as-
sessing their performance in bullish, bearish, and volatile markets. Backtesting results indicate
that the Stockformer-based stock selection model achieves annualized returns of 239.73%, -15.18%,
and 44.48% in rising, falling, and fluctuating markets, respectively; with maximum drawdowns of
3.07%, 19.6%, and 4.14%; and Sharpe ratios of 8.46, -0.56, and 2.71. Compared to other models,
Stockformer displays more stable performance across different market conditions and secures stable
excess returns over the CSI 300 index.
In conclusion, Stockformer demonstrates the potential of deep learning technologies in analyzing
complex markets and provides financial market analysts with a valuable decision-support tool,
aiding them in making wiser investment decisions in volatile market environments.
7.2. Future Work
Despite the significant achievements of the Stockformer model in the application to stock mar-
kets, future research and development are still filled with challenges and opportunities. Firstly,
the wavelet transform used in the model involves multiple complex parameter settings, such as
periodic scales and transformation depths. The determination of these parameters is both time-
consuming and requires repetitive experimentation. To streamline this process, future work will
explore how to automatically identify and optimize these periodic parameters within the neural
network architecture, thereby alleviating the burden of preliminary data processing.
Secondly, the current model is based on a static pool of CSI 300 stocks. However, the dynamic
nature of the stock pool is a fundamental characteristic of stock markets, and using a fixed stock
pool might limit the model’s applicability and predictive accuracy. Therefore, there are plans to
develop a dynamic updating mechanism that allows Stockformer to adapt to changes in the stock
pool, thus enhancing the precision of its predictions.
Additionally, as new stocks are continuously introduced, the model needs to be able to adapt
44

## Page 45

quickly to these changes. To this end, we plan to employ meta-learning approaches to enhance the
adaptability of the model, allowing it to maintain efficient predictive capabilities under unknown
market conditions. With these improvements, Stockformer will be better equipped to serve financial
market analysts, helping them make more precise decisions in a volatile market environment.
Acknowledgements
This research is supported by the National Natural Science Foundation of China under grant
no. 12001556, the National Key Research and Development Program of China under the “National
Key R&D Program of China (no. 2023YFF0614700)”, the Program for Innovation Research in
Central University of Finance and Economics, and the Central University of Finance and Economics
Postgraduate Thesis Competition.
References
A¨ ıt-Sahalia, Y., Fan, J., Xue, L., & Zhou, Y. (2022). How and When are High-Frequency Stock Returns Predictable? .
Technical Report National Bureau of Economic Research.
Ampomah, E. K., Qin, Z., & Nyame, G. (2020). Evaluation of tree-based ensemble machine learning models in
predicting stock price direction of movement. Information ,11, 332.
Ang, A., & Bekaert, G. (2007). Stock return predictability: Is it there? The Review of Financial Studies ,20,
651–707.
Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks
for sequence modeling. arXiv preprint arXiv:1803.01271 , .
Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. The quarterly journal of
economics ,131, 1593–1636.
Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders
and long-short term memory. PloS one ,12, e0180944.
Bollerslev, T., Marrone, J., Xu, L., & Zhou, H. (2014). Stock return predictability and variance risk premia:
Statistical inference and international evidence. Journal of Financial and Quantitative Analysis ,49, 633–661.
Butaru, F., Chen, Q., Clark, B., Das, S., Lo, A. W., & Siddique, A. (2016). Risk and risk management in the credit
card industry. Journal of Banking & Finance ,72, 218–239.
Cai, D., Zhang, T., Han, K., Liang, J. et al. (2022). Economic policy uncertainty shocks and chinese stock market
volatility: An empirical analysis with svar. Complexity ,2022 .
Campbell, J. Y., & Cochrane, J. H. (1999). By force of habit: A consumption-based explanation of aggregate stock
market behavior. Journal of political Economy ,107, 205–251.
Campbell, J. Y., & Thompson, S. B. (2008). Predicting excess stock returns out of sample: Can anything beat the
historical average? The Review of Financial Studies ,21, 1509–1531.
45

## Page 46

Chan, E. P. (2021). Quantitative trading: how to build your own algorithmic trading business . John Wiley & Sons.
Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd
international conference on knowledge discovery and data mining (pp. 785–794).
Cho, K., Van Merri¨ enboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).
Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint
arXiv:1406.1078 , .
Chong, E., Han, C., & Park, F. C. (2017). Deep learning networks for stock market analysis and prediction:
Methodology, data representations, and case studies. Expert Systems with Applications ,83, 187–205.
Cochrane, J. H. (2011). Presidential address: Discount rates. The Journal of finance ,66, 1047–1108.
Dami, S., & Esterabi, M. (2021). Predicting stock returns of tehran exchange using lstm neural network and feature
engineering technique. Multimedia Tools and Applications ,80, 19947–19970.
Daubechies, I. (1992). Ten lectures on wavelets . SIAM.
De Prado, M. L. (2018). Advances in financial machine learning . John Wiley & Sons.
Dorogush, A. V., Ershov, V., & Gulin, A. (2018). Catboost: gradient boosting with categorical features support.
arXiv preprint arXiv:1810.11363 , .
Ehsani, S., & Linnainmaa, J. T. (2022). Factor momentum and the momentum factor. The Journal of Finance ,77,
1877–1919.
Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. the Journal of Finance ,47,
427–465.
Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. Journal of financial economics ,116, 1–22.
Fan, J., Liao, Y., & Wang, M. (2016). Factors that fit the time series and cross-section of stock returns. Journal of
Finance ,71, 2149–2174.
Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predic-
tions. European journal of operational research ,270, 654–669.
Fung, W., & Hsieh, D. A. (1997). Empirical characteristics of dynamic trading strategies: The case of hedge funds.
The review of financial studies ,10, 275–302. doi: 10.1093/rfs/10.2.275 .
Georgakopoulos, H. (2015). Quantitative trading with R: understanding mathematical and computational tools from
a quant’s perspective . Springer.
Green, J., Hand, J. R., & Zhang, X. F. (2017). The characteristics that provide independent information about
average us monthly stock returns. The Review of Financial Studies ,30, 4389–4436.
Gunduz, H. (2021). An efficient stock market prediction model using hybrid feature reduction method based on
variational autoencoders and recursive feature elimination. Financial innovation ,7, 28.
Guo, J., Wang, S., Ni, L. M., & Shum, H.-Y. (2022). Quant 4.0: Engineering quantitative investment with automated,
explainable and knowledge-driven artificial intelligence. arXiv preprint arXiv:2301.04020 , .
Harvey, C. R., Liu, Y., & Zhu, H. (2016). . . . and the cross-section of expected returns. The Review of Financial
Studies ,29, 5–68.
He, Z., & Krishnamurthy, A. (2013). Intermediary asset pricing. American Economic Review ,103, 732–770.
Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation ,9, 1735–1780.
46

## Page 47

Jensen, M. C., Kelly, B. T., & Pedersen, L. H. (2021). Beta-adjusting factor returns. Quantitative Finance ,21, 1–20.
Jiang, F. (2011). How predictable is the Chinese stock market? . Singapore Management University (Singapore).
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). Lightgbm: A highly efficient
gradient boosting decision tree. Advances in neural information processing systems ,30.
Li, J., Bu, H., & Wu, J. (2017). Sentiment-aware stock market prediction: A deep learning method. In 2017
international conference on service systems and service management (pp. 1–6). IEEE.
Li, J., & Huang, S. (2021). The dynamic relationship between economic policy uncertainty and substantial economic
growth in china. Marine Economics and Management ,4, 113–134.
Li, T., Ma, F., Zhang, X., & Zhang, Y. (2020). Economic policy uncertainty and the chinese stock market volatility:
Novel evidence. Economic Modelling ,87, 24–33.
Li, Y., Fu, K., Wang, Z., Shahabi, C., Ye, J., & Liu, Y. (2018). Multi-task representation learning for travel time
estimation. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data
mining (pp. 1695–1704).
Li, Z., Yang, D., Zhao, L., Bian, J., Qin, T., & Liu, T.-Y. (2019). Individualized indicator for all. In Proceedings of
the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining . ACM.
Lin, H., Zhou, D., Liu, W., & Bian, J. (2021). Learning multiple stock trading patterns with temporal routing
adaptor and optimal transport. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery &
data mining (pp. 1017–1026).
Muhammad, S., & Ali, G. (2018). The relationship between fundamental analysis and stock returns based on the
panel data analysis; evidence from karachi stock exchange (kse). Research Journal of Finance and Accounting ,9,
2222–2847.
Qian, E. E., Hua, R. H., & Sorensen, E. H. (2007). Quantitative equity portfolio management: modern techniques
and applications . CRC Press.
Qolipour, F., Ghasemzadeh, M., & Mohammad-Karimi, N. (2021). The predictability of tree-based machine learning
algorithms in the big data context. International Journal of Engineering ,34, 82–89.
Schwartz, R. A. (1970). Efficient capital markets: A review of theory and empirical work: Discussion. The Journal
of Finance ,25, 421–423.
Senxin, Z., Sitong, Z., Linna, X. et al. (2023). Research on analysis and application of quantitative investment
strategies based on deep learning. Academic Journal of Computing & Information Science ,6, 24–30.
Sharpe, W. F. (1966). Mutual fund performance. The Journal of business ,39, 119–138.
Sirignano, J., Sadhwani, A., & Giesecke, K. (2016). Deep learning for mortgage risk. arXiv preprint arXiv:1607.02470 ,
.
Song, C.-H., Xiao, X., Zhang, B., & Xia, S.-T. (2023). Follow the will of the market: A context-informed drift-
aware method for stock prediction. In Proceedings of the 32nd ACM International Conference on Information and
Knowledge Management (pp. 2311–2320).
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser,  L., & Polosukhin, I. (2017).
Attention is all you need. Advances in neural information processing systems ,30.
Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., Bengio, Y. et al. (2017). Graph attention networks.
47

## Page 48

stat,1050 , 10–48550.
Wang, J., Zhuang, Z., & Feng, L. (2022). Intelligent optimization based multi-factor deep learning stock selection
model and quantitative trading strategy. Mathematics ,10, 566.
Wang, Q., & Hao, Y. (2020). Alstm: An attention-based long short-term memory framework for knowledge base
reasoning. Neurocomputing ,399, 342–351.
Wang, Y.-C., Tsai, J.-J., & Li, Q. (2017). Policy impact on the chinese stock market: From the 1994 bailout policies
to the 2015 shanghai-hong kong stock connect. International Journal of Financial Studies ,5, 4.
Wang, Z., Zhang, Z., Zhang, Q., Gao, J., & Lin, W. (2021). Covid-19 and financial market response in china: Micro
evidence and possible mechanisms. Plos one ,16, e0256879.
Xuemei, H., & Junwen, Y. (2024). Group penalized multinomial logit models and stock return direction prediction.
IEEE Transactions on Information Theory , .
Yuan, H., Li, G., Bao, Z., & Feng, L. (2020). Effective travel time estimation: When historical trajectories over road
networks matter. In Proceedings of the 2020 acm sigmod international conference on management of data (pp.
2135–2149).
Yue, H., Liu, J., & Zhang, Q. (2022). Applications of markov decision process model and deep learning in quantitative
portfolio management during the covid-19 pandemic. Systems ,10, 146.
Zheng, B., Zhou, D.-W., Ye, H.-J., & Zhan, D.-C. (2023). Preserving locality in vision transformers for class
incremental learning. arXiv preprint arXiv:2304.06971 , .
48

