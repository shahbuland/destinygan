# destinygan
Automatically generating Destiny 2 loot using StyleGAN2 with differntiable augmentation and GPT-3

# Post-mortem  
This is a project I haven't worked on in well over a year and could not complete. I've added to this repo everything I did accomplish for anyone that is interested in looking at the results I did end up getting or wants to make something else using the data scraper/model setup.  
  
# Things I did finish:  
 - A scraper that gets all gun flavor text and names  
 - A scraper that gets all gun images  
  
# Things that were WIP:  
 - Scraper for perk icons  
 - A model made specifically to generate small icons  
 - A contrastive model that made small icons FOR guns  
 - Final generated guns were meant to have their corner icon and a perk pool  
   
# What went wrong?  
I suspect the dataset was not diverse enough for styleGAN to be trained properly. I proceeded to get trash results when using other peoples' implementations as well. The main issue was unavoidable mode collapse. This still led to me being able to generate some cool guns because different models gave different generations, but not having the style vector effect generated images kind of defeats the purpose of stylegan. I ended up just randomly sampling from different checkpoints to get generated sample diversity. 
  
# What went right?  
The samples that did come out of the model still looked cool! Here are some samples: [imgur album](https://imgur.com/a/bs3XfzW). Looking at it train and go from learning basic shapes to the actual weapons was quite cool as well: [timelapse here](https://www.youtube.com/shorts/HFLLiQztvTA). GPT-3 also did an amazing job of generating new names and flavor texts. The model also seemed to work quite well for the small icons: [imgur album](https://imgur.com/a/pSW3NIK) (this is throughout training so they'll get better as you scroll down). This probably further implies the issue WAS with the dataset and not my code.  
  
# Out of repo requirements:  
- For text generation you need a GPT-3 key
- For scraping destiny data you need bungie API key  
- Several of the model scripts require custom cuda kernels from the official stylegan repo from nvidia. Credit to them for that, I've putted it in a zip file for convenience: [link](https://drive.google.com/file/d/1QWH3_jJV65cN3ebc0yZvXJ3t_ROm3ZG3/view?usp=sharing). Just extract into model stylegan folder.  
- The SWAGAN model probably isn't that well known but they reported much faster training so I implemented it here. Paper [here](https://arxiv.org/abs/2102.06108) if you want more info.  

# Select Samples:  
Fun fact: the names it uses that were already names of guns from the game were not in the prompt! This means GPT-3 has already read articles on Destiny and has memory of names of certain weapons from it. Certainly, it never saw No Land Beyond in my dataset, as it consists entirely of D2 guns.  
![image](https://user-images.githubusercontent.com/44281577/159820638-bb0d6025-adb8-4eb7-94e1-42fe96472a77.png)  
The flavor text is pretty funny sometimes.  
![image](https://user-images.githubusercontent.com/44281577/159820710-114e86a4-c748-4012-8e17-021423ae771a.png)
Apparently Cayde is just a guy.  
![image](https://user-images.githubusercontent.com/44281577/159820790-8c5ea22c-11dd-485f-b5f6-cba890d460c1.png)



