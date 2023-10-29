<h1 align="center"><img src="https://i.imgur.com/HsWXQTW.png" width="24px" alt="logo" /> Hotshot-XL + Gradio</h1>


I forked the Hotshot-XL Project because the build instructions didn't work for me and I wanted to write an interface for it.  Here is the result for you to try out. Please comment with any suggestions or fixes, etc.

## Setup
To deploy this run the following commands in WSL or some Linux with CUDA support.  You may want to use a virtual env for python to keep your libraries in order.

***Basic Setup***
```bash
git clone https://github.com/RamboRogers/Hotshot-XL-Gradio
cd Hotshot-XL-Gradio
pip install -r requirements.txt
python3 app.py
```

***Virtual Environment Setup***
```
pip install virtualenv --upgrade
virtualenv -p $(which python3) venv
source venv/bin/activate
cd env
git clone https://github.com/RamboRogers/Hotshot-XL-Gradio
cd Hotshot-XL-Gradio
pip install -r requirements.txt
python3 app.py
```


You should now be able to generate gif files or mp4 files in the web interface by going to [http://127.0.0.1:7860/](http://127.0.0.1:7860/). The preview will populate once you start generating files.

> *⚠️Note: The mp4 files can't be previewed by Gradio and will show an error but they can be downloaed and viewed in the browser!*

![Interface](interface.gif)

# Acknowledgements 
1. [Gradio](https://www.gradio.app/) rocks
2. [Hotshot-XL](https://github.com/hotshotco/Hotshot-XL) this is the source

---

Coded 🧾 by [Matthew Rogers](https://matthewrogers.org) | [RamboRogers](https://github.com/ramboRogers)
