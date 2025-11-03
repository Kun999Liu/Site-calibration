%% 将高光谱反射率Rrs转化为卫星某波段等效遥感反射率
clear;clc;close all
%>>>>>>>>>>>>>>>>>需手动输入下三项<<<<<<<<<<<<<<<<<
inputrrs="ASD高光谱实测Rrs示例数据.xlsx"; %高光谱反射率excel文件名
inputSRF="GOCI2_SRF.xlsx"; %卫星光谱响应函数excel文件名
output="GOCI2等效遥感反射率.xlsx"; %设置输出等效遥感反射率文件名

%% 读取数据
try 
Rrs=readcell(inputrrs);  %读取高光谱遥感反射率Rrs
SRF=readcell(inputSRF);  %读取传感器光谱响应函数SRF
catch
    disp('>>>>>>>读取失败，请检查Rrs与SRF的excel文件名是否有误<<<<<<<')
end

sn=Rrs(1,2:end); %站点名station name
bn=SRF(1,2:end); %波段名band name

Rrs=cell2mat(Rrs(2:end,:));
SRF=cell2mat(SRF(2:end,:));

c1=width(Rrs); %高光谱反射率矩阵大小 
c2=width(SRF); %SRF矩阵大小 

%将光谱响应函数中可能存在的NaN值与小于0的值变为0
SRF(isnan(SRF))=0;
SRF(find(SRF<0))=0;

%% 计算等效Rrs
er=zeros(c2-1,c1-1); %创建存储等效Rrs的矩阵

for i=2:c2
    for j=2:c1
        l1=find(SRF(:,i)~=0,1); %积分下限
        l2=find(SRF(:,i)~=0,1,'last'); %积分上限
        er(i-1,j-1)=(trapz(Rrs(find(Rrs(:,1)==SRF(l1,1)):find(Rrs(:,1)==SRF(l2,1)),j).*SRF(l1:l2,i)))/(trapz(SRF(l1:l2,i)));
    end
end

%% 数据保存
%创建标好站点名和波段名的excel
temp=cell(c2,c1);
temp(2:end,1)=cellstr(bn');
temp(1,2:end)=cellstr(sn);
temp(2:end,2:end)=num2cell(er);
%输出
writecell(temp,output)

