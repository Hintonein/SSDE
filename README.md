# Symbolic Solver for Differential Equations

This is the source code for the paper "Closed-form Solutions: A New Perspective on Solving Differential Equations".

## Install SSDE


```bash
# Intall kernel
conda install jupyter ipykernel
conda create -n ssde python=3.6
conda activate ssde
conda install ipykernel
python -m ipykernel install --user  --name ssde --display-name "ssde"

# Install ssde
pip install --upgrade setuptools pip
# install the necessary dependencies
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install SSDE package and core dependencies
pip install -e ./ssde
```


## Reference
If you use this code in your research, please cite our paper:

```bibtex

@inproceedings{wei2025closed,
  title={Closed-form Solutions: A New Perspective on Solving Differential Equations},
  author={Shu Wei, Yanjie Li, Lina Yu, Weijun Li, Min Wu, Linjun Sun, Jingyi Liu, Hong Qin, Yusong Deng, Jufeng Han, Yan Pang},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```