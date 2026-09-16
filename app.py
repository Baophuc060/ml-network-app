import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import erfc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title='ML Network Configuration Search', page_icon='📡', layout='wide')
FEATURES=['Pt_dBm','Bandwidth_MHz','Distance_km','PathLoss_dB','Noise_dBm']
BER_MAX=1e-3
THROUGHPUT_MIN=4_000_000
N_CANDIDATES=20_000

@st.cache_data
def load_data():
    import csv

    file_path = 'du_lieu_huan_luyen.csv'

    # Đọc file CSV
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        lines = f.readlines()

    # File hiện tại có thể đang bị " bao quanh toàn bộ dòng
    clean_lines = []

    for line in lines:
        line = line.rstrip('\r\n')

        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        clean_lines.append(line)

    # Đọc dữ liệu bằng dấu phẩy
    from io import StringIO

    df = pd.read_csv(
        StringIO('\n'.join(clean_lines)),
        sep=','
    )

    # Xóa khoảng trắng thừa ở tên cột
    df.columns = df.columns.str.strip()

    return df

@st.cache_resource
def train_models():
    df=load_data(); X=df[FEATURES].copy()
    yber=np.log10(df['BER'].clip(lower=1e-300)); ytp=df['Throughput_bps']; yq=df['QoS']
    Xtr,_,ybtr,_,yttr,_,yqtr,_=train_test_split(X,yber,ytp,yq,test_size=.2,random_state=42)
    models=[RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1) for _ in range(3)]
    models[0].fit(Xtr,ybtr); models[1].fit(Xtr,yttr); models[2].fit(Xtr,yqtr)
    return models

def formula(row):
    Bhz=float(row['Bandwidth_MHz'])*1e6; Pr=float(row['Pt_dBm'])-float(row['PathLoss_dB'])
    snr=10**((Pr-float(row['Noise_dBm']))/10); Rb=Bhz/2; ebn0=snr*(Bhz/Rb)
    ber=.5*erfc(np.sqrt(ebn0)); per=1-(1-ber)**1000; tp=Rb*(1-per)
    qos=int(ber<=1e-3 and tp>=.8*Rb)
    return ber,tp,qos

def predict(config,models):
    X=pd.DataFrame([config],columns=FEATURES)
    return float(10**models[0].predict(X)[0]),float(models[1].predict(X)[0]),int(models[2].predict(X)[0]>=.5)

def search(models):
    rng=np.random.default_rng(42)
    c=pd.DataFrame({'Pt_dBm':rng.uniform(15,25,N_CANDIDATES),'Bandwidth_MHz':rng.uniform(5,15,N_CANDIDATES),'Distance_km':rng.uniform(1,3,N_CANDIDATES),'PathLoss_dB':rng.uniform(90,100,N_CANDIDATES),'Noise_dBm':rng.uniform(-95,-85,N_CANDIDATES)})
    c['BER']=10**models[0].predict(c[FEATURES]); c['Throughput_bps']=models[1].predict(c[FEATURES]); c['QoS']=(models[2].predict(c[FEATURES])>=.5).astype(int)
    c=c[(c.BER<=BER_MAX)&(c.Throughput_bps>=THROUGHPUT_MIN)&(c.QoS==1)]
    return c.sort_values(['Pt_dBm','BER','Throughput_bps'],ascending=[True,True,False]).head(10)

st.title('📡 HỆ THỐNG ĐÁNH GIÁ & TÌM KIẾM CẤU HÌNH MẠNG')
st.write('Nhập 5 thông số → ML dự đoán BER, Throughput, QoS → kiểm chứng bằng công thức gốc → tìm cấu hình tham khảo đạt yêu cầu.')
with st.sidebar:
    st.header('⚙️ Thông số đầu vào')
    Pt=st.number_input('Pt – Công suất phát (dBm)',15.,25.,20.,.1)
    B=st.number_input('B – Băng thông (MHz)',5.,15.,10.,.1)
    d=st.number_input('d – Khoảng cách (km)',1.,3.,2.,.1)
    PL=st.number_input('PL – Path Loss (dB)',90.,100.,95.,.1)
    N=st.number_input('N – Công suất nhiễu (dBm)',-95.,-85.,-90.,.1)
    st.divider(); st.write('Điều kiện đạt:'); st.write('• BER ≤ 10⁻³\n• Throughput ≥ 4 Mbps\n• QoS = Đạt')
    run=st.button('🔍 ĐÁNH GIÁ & TÌM CẤU HÌNH',use_container_width=True,type='primary')
try:
    df=load_data(); models=train_models()
except FileNotFoundError:
    st.error('❌ Không tìm thấy du_lieu_huan_luyen.csv. Hãy đặt file CSV cùng thư mục với app.py.'); st.stop()
st.caption(f'Dataset: {len(df):,} mẫu | 5 input | 3 output | Random Forest: 200 cây/mô hình')
if run:
    config={'Pt_dBm':Pt,'Bandwidth_MHz':B,'Distance_km':d,'PathLoss_dB':PL,'Noise_dBm':N}
    with st.spinner('Đang dự đoán và tìm cấu hình tham khảo...'):
        bm,tm,qm=predict(config,models); bf,tf,qf=formula(pd.Series(config)); refs=search(models)
    st.subheader('1️⃣ Cấu hình đang kiểm tra'); cols=st.columns(5)
    for col,label,val,unit in zip(cols,['Pt','B','d','PL','N'],[Pt,B,d,PL,N],['dBm','MHz','km','dB','dBm']): col.metric(label,f'{val:.2f} {unit}')
    st.subheader('2️⃣ Kết quả Machine Learning'); a,b,c=st.columns(3); a.metric('BER dự đoán',f'{bm:.4e}'); b.metric('Throughput dự đoán',f'{tm/1e6:.4f} Mbps'); c.metric('QoS dự đoán','ĐẠT' if qm else 'KHÔNG ĐẠT')
    st.subheader('3️⃣ Kiểm chứng bằng công thức gốc'); a,b,c=st.columns(3); a.metric('BER công thức',f'{bf:.4e}'); b.metric('Throughput công thức',f'{tf/1e6:.4f} Mbps'); c.metric('QoS công thức','ĐẠT' if qf else 'KHÔNG ĐẠT')
    if bf<=BER_MAX and tf>=THROUGHPUT_MIN and qf: st.success('✅ CẤU HÌNH ĐẠT YÊU CẦU')
    else: st.warning('⚠️ CẤU HÌNH CHƯA ĐẠT. Bên dưới là các cấu hình tham khảo có thể đạt.')
    st.subheader('4️⃣ Các cấu hình tham khảo có thể đạt')
    if refs.empty: st.info('Chưa tìm thấy cấu hình đạt trong số mẫu tìm kiếm.')
    else:
        out=refs[FEATURES+['BER','Throughput_bps','QoS']].copy(); out['BER']=out.BER.map(lambda x:f'{x:.3e}'); out['Throughput_bps']=(out.Throughput_bps/1e6).map(lambda x:f'{x:.4f} Mbps'); out['QoS']=out.QoS.map(lambda x:'ĐẠT' if x else 'KHÔNG ĐẠT'); out.columns=['Pt (dBm)','B (MHz)','d (km)','PL (dB)','N (dBm)','BER','Throughput','QoS']; st.dataframe(out,use_container_width=True,hide_index=True)
        st.caption(f'Đã quét {N_CANDIDATES:,} cấu hình trong đúng phạm vi dataset. Đây là cấu hình tham khảo, không khẳng định tối ưu toàn cục.')
st.divider(); st.caption('Lưu ý: d được giữ đúng là input của mô hình theo project hiện tại; công thức gốc hiện không trực tiếp dùng d để tính BER/Throughput.')
