
import streamlit as st
import pandas as pd
import pickle
from PIL import Image

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

    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            In Epitech, students are facing with their deepest nightmare - the gender gap in Tech. With less than 3% of women per class (in average), Epitech students have no way to practice the courtship practices so common at other schools. By the time they leave the school, the students are either unfit for the other gender, or at best have become monks with a vow of abstinence.    
            However, our group is here to save the day. With our Cyber Waifu Generator, students can now create the perfect woman to talk with, without the fear of being judged.
            """)
        '''
        ## How does it work ❓ 
        Click and Enjoy!
        '''

    with col2:
        st.subheader("Make the perfect waifu appear")

        ## TODO image placeholder

        if (st.button("Enjoy!")):
            col1.write('''
                ## Processing... ⌛            
            ''')
            col2.success('''
                CYBER WAIFUUUUU
            ''')

    st.markdown(footer, unsafe_allow_html=True)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

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
