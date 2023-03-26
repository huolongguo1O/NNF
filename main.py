import gradio as gr


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import mkldnn

import matplotlib.pyplot as plt
from torch.autograd import Variable
import random
from tqdm import tqdm
from rich.progress import track
from cmath import nan
import numpy
plt.switch_backend('agg')
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.hidden3 = nn.Linear(n_hidden,n_hidden)
        self.hidden4 = nn.Linear(n_hidden,n_hidden)
        self.hidden5 = nn.Linear(n_hidden,n_hidden)
        self.hidden6 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.hidden4(out)
        out = F.relu(out)
        out = self.hidden5(out)
        out = F.relu(out)
        out = self.hidden6(out)
        out = F.relu(out)
        out =self.predict(out)

        return out
        

def train(data,lr,epochs,n_hidden,fn_name):
    x=[]
    y=[]
    _x=[]
    _t=[]


    lines=data.split('\n')
    for i in lines:
        print(i)
        x.append(i.split('  ')[0])
        y.append(float(i.split('  ')[1].replace('\n','')))



    for i in x:
        t=i.split(' ')
        print(t)
        _t=[]
        for ix in t:
            _t.append(float(ix))
        _x.append(_t)

    x = torch.Tensor(_x)

    y = torch.Tensor(y)
    y=torch.unsqueeze(y,dim=1)

    x , y =(Variable(x),Variable(y))
    fig, ax = plt.subplots()



    net = Net(len(_x[0]),n_hidden,1)


    print(net)
    print(net.parameters())
    lastnet=net

    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
    loss_func = torch.nn.MSELoss()
    _t=[]
    _loss=[]
    # plt.ion()
    # plt.show()
    lastloss=0
    print('Preparing:')
    for t in track(range(100)):
        prediction = net(x)
        loss = loss_func(prediction,y)
        loss.backward()
        optimizer.step()
    for t in track(range(epochs)):
        prediction = net(x)
        loss = loss_func(prediction,y)
        if numpy.isnan(loss.data):
            return '学习率太大力'
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if t%50 ==0:

            # plt.cla()

            _t.append(t)
            _loss.append(loss.data)
            
            lastloss=loss.data
            
            lastnet=net
            # plt.pause(0.0000005)
    fig=plt.figure()
    plt.plot(_t,_loss,'r-')
    plt.text(0 , 0, 'Loss = %.5f lr=%f' % (loss.data,lr), fontdict={'size': 10, 'color': 'blue'},transform=ax.transAxes)
    print('--End--')
    # plt.close()
    # torch.save()
    torch.save(net,fn_name+'.pkl')
    # torch.save(net.state_dict(),'net_parameter.pkl')
    return fig

def predict(data,netpkl):
    d=[]
    _d=[]
    __d=[]
    # print(netpkl)
    # netpkl.save()
    net1 = torch.load(netpkl+'.pkl')
    d=data.split('\n')
    ret=''
    for i in d:
        _d=i.split(' ')
        # print(t)
        _t=[]
        for ix in _d:
            _t.append(float(ix))
        __d.append(_t)
    predict=net1(torch.Tensor(__d))
    # convert predict to numpy
    predict=predict.detach().numpy()

    for i in predict:
        ret=ret+str(i)+' '
    return ret

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
with gr.Blocks() as demo:
    with gr.Tab("训练"):
        data = gr.Textbox(label="数据（例如 1 1  2）",lines=10,max_lines=10000000000000000,value='1 1  2\n2 3  5')
        lr = gr.Slider(label="学习率",minimum=0.000000001,maximum=1,value=0.0001,step=0.000000001)
        epochs = gr.Slider(label="训练轮次",minimum=1000,maximum=100000,value=5000)
        n_hidden = gr.Slider(label="隐藏层大小",minimum=1,maximum=1000,value=128)
        fn_name = gr.Textbox(label="函数名")
        # output = gr.File(label="模型")
        output=gr.Plot()
        greet_btn = gr.Button("Go!")
        greet_btn.click(fn=train, inputs=[data,lr,epochs,n_hidden,fn_name], outputs=output)
    with gr.Tab("预测"):
        data = gr.Textbox(label="数据（例如 1 1）",lines=1,max_lines=10000000000000000,value='1 1')
        netpkl = gr.Textbox(label="函数名")
        output = gr.Textbox(label='Output')
        greet_btn = gr.Button("Go!")

        greet_btn.click(fn=predict, inputs=[data,netpkl], outputs=output)
    gr.Markdown('''
# NNF(神经网络函数)
###### 用神经网络实现的函数，可用于黑盒预测。
## 1.训练
数据采用`输入1 输入2 输入3……（两个空格）输出` 形式 。
函数名一定要记住且不重复（否则会覆盖）。
其余参数建议默认。
接着点击训练即可。
## 2.预测
数据为`输入1 输入2 输入3……`
函数名即训练时填写的函数名。
    ''')
demo.launch(share=True)   
