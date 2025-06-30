# Causal-VidSyn-Office
Egocentricly comprehending the causes and effects of car accidents is crucial for the safety of self-driving cars, and synthesizing causal-entity reflected accident videos can facilitate the capability test to respond to unaffordable accidents in reality. However, incorporating causal relations as seen in real-world videos into synthetic videos remains challenging. This work argues that precisely identifying the accident participants and capturing their related behaviors are of critical importance. In this regard, we propose a novel diffusion model Causal-VidSyn for synthesizing egocentric traffic accident videos. To enable causal entity grounding in video diffusion, Causal-VidSyn leverages the cause descriptions and driver fixations to identify the accident participants and behaviors, facilitated by accident reason answering and gaze-conditioned selection modules. To support CausalVidSyn, we further construct Drive-Gaze, the largest driver gaze dataset (with 1.54M frames of fixations) in driving accident scenarios. Extensive experiments show that CausalVidSyn surpasses state-of-the-art video diffusion models in terms of frame quality and causal sensitivity in various tasks, including accident video editing, normal-to-accident video diffusion, and text-to-video generation.

## DATA Download
a.Download MM-AU Dataset. You can log in to our [project homepage](http://www.lotvsmmau.net) to download the benchmark.

Make a data structure
>[rootpath]
>>[CAP-DATA (MM-AU)]
>>>[trainxx.txt]

b.Download [Drive-Gaze and train.txt](https://pan.baidu.com/s/1FWgrNmK2hfAv9VH3sEWZqA?pwd=u9es).

c.Download the Pre-trained video diffusion models [here](https://pan.baidu.com/s/1eORCcoWz7hWRIGJd9Wy3nA?pwd=4g4i)

d.Download the [QA dataset](https://pan.baidu.com/s/1j0PpptGEO0F7lh_PevkMfw?pwd=i2xk ) in MM-AU Training

## Training
a.You can use our MM-AU Dataset and load the initial Pre-trained video diffusion models.

```Run train_stage0.py```

b.Use the checkpoint saved in stage 0.

```Run train_stage1.py```

b.Use the checkpoint saved in stage 1.

```Run train_stage2.py```

## Inference
a.You can download our trained model [here](https://pan.baidu.com/s/1lSYrJGTvRAqrfDIkGUlGGQ?pwd=d9cx)

b.Use the V2VPipeline we provide during the training stage.

## Evaluation
a.You can see the ```eval_metrics.py``` to calculate CLIP Score, FVD, and Temporal Consistency.

b. Additionally, you can choose the master's branch in this project to calculate Affordance using GroundDINO. 
The weights of GroundDINO you can download [here](https://pan.baidu.com/s/1AAN7-VDaJ5UsWczEEwwHMQ?pwd=39h2) 
