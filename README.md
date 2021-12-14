
<style>
    h1 {
        text-align: center; color: black; font-style: bold; background-color:DARKORANGE; border-radius: 25px;
    }
    h2 {
        color: Cyan;
    }
    h3 {
        color: skyblue;
        font-style: italic;
    }

    h4 {
        text-align: center;
        background-color: Darkorange; border-radius: 5px;
    }

    h4 > a {
        font-size: 5vh;
        color: black;
    }
</style>


# WAIFU GENERATOR

## Contributors
Epitech 4th year students - Global Nomad Track Program
- [Aldric Liottier](https://github.com/aldricLiottier)
- [Léo "softhy85" Ménard](https://github.com/softhy85)
- [Tim "Tim-Snugget" Fertin](https://github.com/Tim-Snugget)

## The Context
This Github is the code we used, created and love during the Epitech Tek4 - Deep Learning module.  
The project goal was to develop and train an AI using Deep Learning techs and deploy a quick Web App displaying the results obtained.  
Our ambition was to be able to generate an OnlyFans model using Deepface and Image Generation technologies. Images we could then use to open the first Cyber Waifu OnlyFans account and generate profits to pay our loan to the banks.  

## The Project
        In Epitech, students are facing with their deepest nightmare - the gender gap in Tech. With less than 3% of women per class (in average), Epitech students have no way to practice the courtship practices so common at other schools. By the time they leave the school, the students are either unfit for the other gender, or at best have become monks with a vow of abstinence.
        However, our group is here to save the day. With our Cyber Waifu Generator, students can now create the perfect woman to talk with, without the fear of being judged.

## The Tech
### The Datasets
The datasets we used were [Human Faces](https://www.kaggle.com/ashwingupta3012/human-faces) *by ashwingupta3012* and [Celeba-Dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) *by jessicali9530*. We then sorted them to only keep the Women from each dataset, giving us a total of 120,821 women's faces' pictures.

### The algorithm
We experimented using the lstm technology, but the result was not that satisfying - inconclusive results mainly, but also technical issues. We then chose to apply the [GAN algorithm](https://en.wikipedia.org/wiki/Generative_adversarial_network).  
It consists of using two models, a generator and a discriminator. The Generator's purpose is to create output from random noise without knowing if the result will be true or false, while the Discriminator's is to identify the false output from the dataset given and the true output. So, during the process, the generator model will have a better quality and the more the AI train, the better the output will be.

### Other
We trained the AI on a local machine (Win 10 - Ryzen 5 1600) using a GPU (GTX 1050 2GB) but encountered difficulty regarding the computing time, memory allocation and time consumption. Indeed, we first wanted to do 60 epochs as an arbitrary value of a well-trained AI. Nonetheless, we then realized that an epoch was estimated to a 90 minutes computing time (in the best scenario). Realising we'd already have our diploms by the time it ended, we then chose to reduce epochs amount to 10. Then again, memory allocations failed, resulting in crash, errors and pauses. We finally put 1 epoch in the delivery to have something to show our wonderful and awesome and unbelievably cool teacher.

## The Result

Since the AI is not really fully trained (yet?) the result is quite inconclusive. Yet, it shows it is definitely achievable... at least, face-wise.  
Indeed, since we only have datasets of faces available, we only focused on said body part, instead of the whole package.  
Generated images are then ***available on our Web App***.

#### [The  Web App is available here](https://share.streamlit.io/tim-snugget/deeplearning_weneedtogodeeper/main).

## Tools
- Python
    - keras
    - Deepface
    - tensorflow
    - and many more...
- Google Collab
- Kaggle
- Discord
- EduFlow
- SaturdaysAI

## References
- [But what is a neural network](https://www.youtube.com/watch?v=aircAruvnKk) *by 3Blue1Brown*
- [Pornhub](https://youtube.com/watch?v=dQw4w9WgXcQ)
- [4chan](ww.4chan.org)
- [OnlyFans](onlyfans.com)
