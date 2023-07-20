from gtts import gTTS
import streamlit as st
from io import BytesIO
from pydub import AudioSegment

from simpleplayer import simpleplayer

player = simpleplayer('trans.mp3')
player.play()
# st_player("trans.mp3")


