
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import os

PATH = "./"
model_dir = "./model"


def load_model(model_file):
    return (model_file, pickle.load(open(f"{model_dir}/{model_file}.pkl", "rb")))


def main():
    # title
    html_temp = """
    <div>
    <h1 style="text-align: center; color: black; text-height: 80%; background-color:DARKORANGE; border-radius: 25px; margin: 5%;">
    Cyber Waifu Face Generator
    </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    with st.expander(" ℹ️ Information", expanded=True):
        st.write("""
        In Epitech, students are facing with their deepest nightmare - the gender gap in Tech. With less than 3% of women per class (in average), Epitech students have no way to practice the courtship practices so common at other schools. By the time they leave the school, the students are either unfit for the other gender, or at best have become monks with a vow of abstinence.\n
        However, our group is here to save the day. With our Cyber Waifu Generator, students can now create the perfect woman to talk with, without the fear of being judged.
        """)
    '''
    ## How does it work ❓
    Select the Epoch you'd like to display, Click and Enjoy!
    '''

    st.subheader("Make the perfect waifu appear")

    ## TODO image placeholder
    epoch_selector = st.select_slider("Epoch", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])

    with c1:
        st.write("")

    with c2:
        st.write("")

    with c3:
        st.write("")

    with c4:
        st.write("")

    with c5:
        st.write("")

    def display_image_grids(epoch):
        images = []
        for i in range(10):
            images.append(Image.open("generated_img_%03d_%d.png" % (epoch - 1, i)))

        c1.image(images[0])
        c1.image(images[5])
        c2.image(images[1])
        c2.image(images[6])
        c3.image(images[2])
        c3.image(images[7])
        c4.image(images[3])
        c4.image(images[8])
        c5.image(images[4])
        c5.image(images[9])

    if (st.button("Enjoy!")):
        st.write('''
            ## Processing... ⌛
        ''')
        if not os.path.exists("generated_img_%03d_0.png" % (epoch_selector - 1)):
            st.error(f"Epoch {epoch_selector} is not avaiable... yet!")
            return
        display_image_grids(epoch_selector)
        st.success('''
            CYBER WAIFUUUUU
        ''')

    st.markdown(footer, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
#     col1, col2 = st.columns([5, 2])
#
#     with col1:
#         with st.expander(" ℹ️ Information", expanded=True):
#             st.write("""
#             In Epitech, students are facing with their deepest nightmare - the gender gap in Tech. With less than 3% of women per class (in average), Epitech students have no way to practice the courtship practices so common at other schools. By the time they leave the school, the students are either unfit for the other gender, or at best have become monks with a vow of abstinence.\n
#             However, our group is here to save the day. With our Cyber Waifu Generator, students can now create the perfect woman to talk with, without the fear of being judged.
#             """)
#         '''
#         ## How does it work ❓
#         Click and Enjoy!
#         '''
#
#     with col2:
#         st.subheader("Make the perfect waifu appear")
#
#         ## TODO image placeholder
#
#         if (st.button("Enjoy!")):
#             col1.write('''
#                 ## Processing... ⌛
#             ''')
#             col2.success('''
#                 CYBER WAIFUUUUU
#             ''')
#
#     st.markdown(footer, unsafe_allow_html=True)
#
# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
height: 6%;
background-color: DARKORANGE;
color: black;
text-align: center;
}


</style>
<div class="footer">
<p>This WebApp is part of a school project, <a href="https://github.com/Tim-Snugget/DeepLearning_WeNeedToGoDeeper">here's the GitHub</a> with all the code and resources available.
Made with <a href="https://streamlit.io">Streamlit</a>.
</p>

</div>
"""


if __name__ == '__main__':
    main()
