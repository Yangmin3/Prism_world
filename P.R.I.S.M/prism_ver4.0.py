import streamlit as st
import pandas as pd
import numpy as np
import requests
import random
from PIL import Image
import torch
import clip
import os
import tempfile
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go
from streamlit_mic_recorder import mic_recorder

# 1. 초기 설정 (전역 실행)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
client = OpenAI(api_key=api_key) if "YOUR_API_KEY" not in api_key and api_key else None

device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="P.R.I.S.M. Final", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")

# 2. 스타일 및 UI 개선 함수
def add_custom_css():
    st.markdown("""
        <style>
            /* 사이드바의 자동 페이지 목록 숨기기 */
            div[data-testid="stSidebarNav"] {
                display: none;
            }
            /* 사이드바 색상을 메인 배경과 동일하게 변경 */
            [data-testid="stSidebar"] {
                background-color: #010c14;
                border-right: 1px solid #00aaff;
            }
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
            .stApp { background-color: #010c14; color: #ffffff; }
            h1, h2, h3, h4, strong, p, div, span, li, label {
                color: #00ffff !important;
                text-shadow: 0 0 3px #00ffff;
            }
            .main-title {
                font-family: 'Orbitron', sans-serif;
                font-size: 52px;
                text-align: center;
                text-shadow: 0 0 5px #00ffff, 0 0 15px #00ffff, 0 0 25px #00aaff;
            }
            .stTabs [data-baseweb="tab"] {
                font-family: 'Orbitron', sans-serif;
                font-size: 18px;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #00aaff;
                box-shadow: 0 2px 10px rgba(0, 170, 255, 0.7);
            }
            /* 예외: 특정 요소들은 그림자/색상 제거로 가독성 확보 */
            .stAlert p, .stAlert .st-emotion-cache-1kyxreq, .st-emotion-cache-1kyxreq.e1f1d6gn0, .stAlert .st-info, .stAlert a {
                color: #010c14 !important;
                text-shadow: none;
            }
            /* 포켓몬 스탯 카드 디자인 */
            .pokemon-stat-card {
                background: linear-gradient(45deg, #052d49, #0f4c75);
                border: 2px solid #3282b8;
                border-radius: 20px;
                padding: 20px;
                margin-top: 20px;
                box-shadow: 0 0 20px rgba(50, 130, 184, 0.7);
            }
            .pokemon-stat-card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #bbe1fa;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .flavor-text {
                font-style: italic;
                opacity: 0.9;
                text-align: center;
                padding-bottom: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

# 3. 데이터 및 모델 로딩 함수
@st.cache_resource(show_spinner="Connecting to P.R.I.S.M. core...")
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_data(show_spinner="Downloading universal Pokémon data...")
def fetch_and_process_pokemon_data():
    MASTER_CSV = "pokemon_data_master.csv"
    if os.path.exists(MASTER_CSV): 
        return pd.read_csv(MASTER_CSV)
    
    type_korean = {
        "normal":"노말", "fire":"불꽃", "water":"물", "electric":"전기", "grass":"풀", 
        "ice":"얼음", "fighting":"격투", "poison":"독", "ground":"땅", "flying":"비행", 
        "psychic":"에스퍼", "bug":"벌레", "rock":"바위", "ghost":"고스트", "dark":"악", 
        "dragon":"드래곤", "steel":"강철", "fairy":"페어리"
    }
    stat_korean = {
        "hp":"HP", "attack":"공격", "defense":"방어", 
        "special-attack":"특수공격", "special-defense":"특수방어", "speed":"스피드"
    }
    
    all_pokemon_data = []
    generation_ranges = {
        1:(1,151), 2:(152,251), 3:(252,386), 4:(387,493), 5:(494,649),
        6:(650,721), 7:(722,809), 8:(810,905), 9:(906,1025)
    }
    
    progress_bar = st.progress(0, text="Synchronizing Pokémon data across dimensions...")
    total_pokemon = generation_ranges[9][1]
    
    for i in range(1, total_pokemon + 1):
        try:
            res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{i}")
            res.raise_for_status()
            pokemon_data = res.json()
            
            species_res = requests.get(pokemon_data['species']['url'])
            species_res.raise_for_status()
            species_data = species_res.json()
            
            ko_name = next((n['name'] for n in species_data['names'] if n['language']['name'] == 'ko'), pokemon_data['name'])
            flavor_text = next((ft['flavor_text'].replace("\n", " ").replace("\x0c", " ") for ft in species_data['flavor_text_entries'] if ft['language']['name'] == 'ko'), "No Korean description available.")
            
            types = [type_korean.get(t['type']['name'], t['type']['name']) for t in pokemon_data['types']]
            stats = {stat_korean.get(s['stat']['name'], s['stat']['name']): s['base_stat'] for s in pokemon_data['stats']}
            
            abilities = []
            for a in pokemon_data['abilities']:
                ability_res = requests.get(a['ability']['url']).json()
                ko_ability = next((n['name'] for n in ability_res['names'] if n['language']['name'] == 'ko'), a['ability']['name'])
                abilities.append(ko_ability)
            
            gen = next((g for g, r in generation_ranges.items() if r[0] <= i <= r[1]), None)
            
            all_pokemon_data.append({
                'id': pokemon_data['id'],
                'name_en': pokemon_data['name'],
                'name_ko': ko_name,
                'image_url': pokemon_data['sprites']['other']['official-artwork']['front_default'],
                'types': ", ".join(types),
                'height': pokemon_data['height']/10.0,
                'weight': pokemon_data['weight']/10.0,
                'abilities': ", ".join(abilities),
                'stats_hp': stats.get('HP',0),
                'stats_attack': stats.get('공격',0),
                'stats_defense': stats.get('방어',0),
                'stats_sp_attack': stats.get('특수공격',0),
                'stats_sp_defense': stats.get('특수방어',0),
                'stats_speed': stats.get('스피드',0),
                'flavor_text': flavor_text,
                'color': species_data['color']['name'],
                'generation': gen
            })
            
            progress_bar.progress(i / total_pokemon, text=f"Synchronizing... {ko_name} ({i}/{total_pokemon})")
        except: 
            continue
    
    df = pd.DataFrame(all_pokemon_data)
    df.to_csv(MASTER_CSV, index=False)
    progress_bar.empty()
    return df

@st.cache_data(show_spinner=False)
def create_clip_embeddings(_df, _model, _preprocess):
    EMBEDDINGS_FILE = "embeddings.npy"
    if os.path.exists(EMBEDDINGS_FILE):
        st.info("기존 분석 패턴(임베딩)을 로드합니다.")
        return np.load(EMBEDDINGS_FILE)
    
    st.info("새로운 시냅스 공명 패턴을 분석(임베딩 생성)합니다...")
    embeddings = []
    progress_bar = st.progress(0, text="Analyzing synaptic patterns...")
    
    for i, row in enumerate(_df.itertuples()):
        feature_text = f"이 포켓몬은 {row.types} 타입이며, 색상은 {row.color}입니다. 설명: {row.flavor_text}"
        text_tokens = clip.tokenize([feature_text], truncate=True).to(device)
        
        with torch.no_grad(): 
            text_features = _model.encode_text(text_tokens)
        
        try:
            response = requests.get(row.image_url, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_input = _preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad(): 
                image_features = _model.encode_image(image_input)
        except: 
            image_features = torch.zeros_like(text_features)
        
        combined_features = (text_features + image_features) / 2
        combined_features /= combined_features.norm(dim=-1, keepdim=True)
        embeddings.append(combined_features.cpu().numpy().flatten())
        
        progress_bar.progress((i + 1) / len(_df), text=f"Analyzing... {row.name_ko}")
    
    np_embeddings = np.array(embeddings)
    np.save(EMBEDDINGS_FILE, np_embeddings)
    progress_bar.empty()
    return np_embeddings

# 4. UI 컴포넌트 함수들
def draw_sidebar_one(df):
    st.sidebar.markdown("<h2>탐색기 제어</h2>", unsafe_allow_html=True)
    gen_options = [f"{i}세대" for i in range(1, 10)]
    gen_choice = st.sidebar.radio("세대별 필터링:", gen_options, horizontal=True)
    gen_filter = int(gen_choice.replace("세대", ""))
    search_query = st.sidebar.text_input("포켓몬 검색 (이름, 타입, 특성):")
    
    filtered_df = df.copy()
    if gen_filter: 
        filtered_df = filtered_df[filtered_df['generation'] == gen_filter]
    if search_query:
        query = search_query.lower()
        filtered_df = filtered_df[
            filtered_df['name_ko'].str.lower().str.contains(query) | 
            filtered_df['types'].str.lower().str.contains(query) | 
            filtered_df['abilities'].str.lower().str.contains(query)
        ]
    return filtered_df

def draw_sidebar_two(df):
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h2>차원 접속기</h2>", unsafe_allow_html=True)
    
    if client is None:
        st.sidebar.warning("OpenAI API 키를 설정하면\n오늘의 운세 기능을 사용할 수 있습니다.")
        return
    
    if st.sidebar.button("오늘의 파트너 포켓몬 소환"):
        st.session_state.todays_pokemon = df.sample(1).iloc[0]
        st.session_state.todays_fortune = None
    
    if 'todays_pokemon' in st.session_state:
        p = st.session_state.todays_pokemon
        st.sidebar.markdown(f"<div class='pokemon-card' style='border-color: #00ffff; text-align: left;'>", unsafe_allow_html=True)
        st.sidebar.image(p['image_url'], caption=f"No.{p['id']} {p['name_ko']}")
        
        if 'todays_fortune' not in st.session_state or st.session_state.todays_fortune is None:
            with st.spinner(f"{p['name_ko']}의 기운을 해석하는 중..."):
                prompt = f"'{p['name_ko']}' 포켓몬을 '오늘의 타로 카드'처럼 생각하고, '[{p['name_ko']}] - OOO 카드' 라는 제목과 함께 오늘 하루의 이운, 관계운, 재물운에 대한 깊이 있는 조언을 신비로운 타로 마스터 스타일로 작성해주세요."
                response = client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[{"role": "user", "content": prompt}]
                )
                st.session_state.todays_fortune = response.choices[0].message.content
        
        st.sidebar.markdown(st.session_state.todays_fortune)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

def draw_pokemon_card(p_data):
    st.markdown(f"""
        <div class="pokemon-stat-card">
            <div class="pokemon-stat-card-header">
                <h3>{p_data.name_ko}</h3>
                <h3>HP: {p_data.stats_hp}</h3>
            </div>
            <p class="flavor-text">"{p_data.flavor_text}"</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.image(p_data.image_url)
            st.markdown(f"**타입:** {p_data.types}")
            st.markdown(f"**특성:** {p_data.abilities}")
            st.markdown(f"**키:** {p_data.height} m | **몸무게:** {p_data.weight} kg")
        
        with col2:
            stats_categories = ['HP', '공격', '방어', '특수공격', '특수방어', '스피드']
            stats_values = [p_data.stats_hp, p_data.stats_attack, p_data.stats_defense, 
                          p_data.stats_sp_attack, p_data.stats_sp_defense, p_data.stats_speed]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=stats_values, 
                theta=stats_categories, 
                fill='toself', 
                name=p_data.name_ko, 
                line_color='#00ffff', 
                fillcolor='rgba(0, 255, 255, 0.4)'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 200]), 
                    bgcolor="rgba(0,0,0,0)", 
                    angularaxis=dict(color="white", showline=False)
                ), 
                paper_bgcolor="rgba(0,0,0,0)", 
                showlegend=False, 
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)

def draw_pokedex_view(filtered_df):
    if st.session_state.get('detail_view_pokemon') is not None:
        draw_pokemon_card(st.session_state.detail_view_pokemon)
        if st.button("목록으로 돌아가기"):
            st.session_state.detail_view_pokemon = None
            st.rerun()
    else:
        st.header("P.R.I.S.M. 데이터베이스")
        if 'detail_view_pokemon' not in st.session_state:
            st.session_state.detail_view_pokemon = None
        
        cols = st.columns(5)
        sorted_df = filtered_df.sort_values('id').reset_index(drop=True)
        
        for i, row in enumerate(sorted_df.itertuples(index=False)):
            col = cols[i % 5]
            with col.container():
                st.image(row.image_url, use_container_width=True)
                st.markdown(f"<h4 style='text-align:center;'>{row.name_ko}</h4>", unsafe_allow_html=True)
                with st.expander("간단 분석"):
                    st.markdown(f"**타입:** {row.types}")
                    st.markdown(f"**특성:** {row.abilities}")
                if st.button("상세 스탯 분석", key=f"detail_{row.id}", use_container_width=True):
                    st.session_state.detail_view_pokemon = row
                    st.rerun()

def draw_resonance_analyzer_view(df, embeddings):
    st.header("시냅스 공명 분석기")
    target_name = st.text_input("분석할 포켓몬 이름을 직접 입력하세요:")
    
    if target_name:
        matches = df[df['name_ko'].str.contains(target_name, case=False)]
        if not matches.empty:
            target = matches.iloc[0]
            target_embedding = embeddings[target.name]
            
            with st.spinner("최고 공명체를 찾는 중..."):
                similarities = cosine_similarity([target_embedding], embeddings)[0]
                similarities[target.name] = -1
                resonant_idx = np.argmax(similarities)
                resonance_score = similarities[resonant_idx]
                resonant_pokemon = df.iloc[resonant_idx]
            
            c1, c2, c3 = st.columns([2, 1, 2])
            
            with c1:
                st.image(target['image_url'], caption=target['name_ko'])
                st.info(f"**타입:** {target['types']}\n\n**특성:** {target['abilities']}\n\n**설명:** {target['flavor_text']}")
            
            with c2:
                st.markdown("<div style='text-align: center; margin-top: 150px;'>", unsafe_allow_html=True)
                st.metric(label="시냅스 공명률", value=f"{resonance_score * 100:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with c3:
                st.image(resonant_pokemon['image_url'], caption=resonant_pokemon['name_ko'])
                st.info(f"**타입:** {resonant_pokemon['types']}\n\n**특성:** {resonant_pokemon['abilities']}\n\n**설명:** {resonant_pokemon['flavor_text']}")
        else:
            st.warning("해당 이름의 포켓몬이 없습니다.")

def draw_analogy_game_view(df, embeddings):
    st.header("포켓몬 관계 유추 게임")
    
    if client is None: 
        st.warning("OpenAI API 키를 설정하면 이 기능을 사용할 수 있습니다.")
        return
    
    if st.button("새로운 문제 생성"):
        a_idx, b_idx, c_idx = df.sample(3).index.tolist()
        target_vec = embeddings[b_idx] - embeddings[a_idx] + embeddings[c_idx]
        similarities = cosine_similarity([target_vec], embeddings)[0]
        
        for idx in [a_idx, b_idx, c_idx]: 
            similarities[idx] = -1
        
        answer_idx = np.argmax(similarities)
        options_idx = list(np.argsort(similarities)[-4:])
        if answer_idx not in options_idx: 
            options_idx.append(answer_idx)
        random.shuffle(options_idx)
        
        st.session_state.analogy_question = (a_idx, b_idx, c_idx, answer_idx, options_idx)
        st.session_state.analogy_answered = False
    
    if 'analogy_question' in st.session_state and st.session_state.analogy_question:
        a_idx, b_idx, c_idx, answer_idx, options_idx = st.session_state.analogy_question
        poke_a, poke_b, poke_c = df.iloc[a_idx], df.iloc[b_idx], df.iloc[c_idx]
        
        sim_ab = cosine_similarity([embeddings[a_idx]], [embeddings[b_idx]])[0][0] * 100
        sim_ac = cosine_similarity([embeddings[a_idx]], [embeddings[c_idx]])[0][0] * 100
        sim_bc = cosine_similarity([embeddings[b_idx]], [embeddings[c_idx]])[0][0] * 100
        
        st.markdown(f"**A-B 관계도:** `{sim_ab:.1f}%` | **A-C 관계도:** `{sim_ac:.1f}%` | **B-C 관계도:** `{sim_bc:.1f}%`")
        
        cols = st.columns(4)
        cols[0].image(poke_a['image_url'], caption=f"A: {poke_a['name_ko']}")
        cols[1].image(poke_b['image_url'], caption=f"B: {poke_b['name_ko']}")
        cols[2].image(poke_c['image_url'], caption=f"C: {poke_c['name_ko']}")
        cols[3].image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/poke-ball.png", caption="D: ???")
        
        if not st.session_state.get('analogy_answered', False):
            options_df = df.iloc[options_idx]
            st.write("---")
            st.write("### 정답은?")
            cols = st.columns(len(options_df))
            
            for idx, col in enumerate(cols):
                with col:
                    st.image(options_df.iloc[idx]['image_url'])
                    if st.button(options_df.iloc[idx]['name_ko'], key=f"option_{idx}", use_container_width=True):
                        st.session_state.analogy_choice = options_df.iloc[idx]['name_ko']
                        st.session_state.analogy_answered = True
                        st.rerun()
        
        if st.session_state.get('analogy_answered', False):
            answer_row = df.iloc[answer_idx]
            st.markdown("---")
            
            if st.session_state.analogy_choice == answer_row['name_ko']: 
                st.success(f"정답입니다! 정답은 **{answer_row['name_ko']}** 입니다.")
            else: 
                st.error(f"아쉽네요. 정답은 **{answer_row['name_ko']}** 입니다.")
            
            st.image(answer_row['image_url'], caption=f"정답: {answer_row['name_ko']}", width=200)
            st.info(f"**타입:** {answer_row['types']}\n\n**특성:** {answer_row['abilities']}")

def draw_20q_game_view(df):
    st.header("스무고개 AI")
    
    if client is None: 
        st.warning("OpenAI API 키를 설정하면 이 기능을 사용할 수 있습니다.")
        return
    
    if 'twentyq_history' not in st.session_state: 
        st.session_state.twentyq_history = []
    
    input_feat = st.text_area("힌트 입력", placeholder="예: 주황색이고 꼬리에 불이 붙어있어", height=100)
    
    col1, col2 = st.columns(2)
    
    if col1.button("힌트 주기") and input_feat:
        st.session_state.twentyq_history.append(input_feat)
        st.rerun()
    
    if col2.button("새 게임 (힌트 초기화)"):
        st.session_state.twentyq_history = []
        st.rerun()
    
    if st.session_state.twentyq_history:
        full_text = ' '.join(st.session_state.twentyq_history)
        st.write(f"지금까지의 힌트: **{full_text}**")
        
        with st.spinner('포켓몬 탐정이 추리 중...'):
            prompt = f"당신은 최고의 포켓몬 탐정이다. 주어진 힌트들을 종합하여 어떤 포켓몬인지 맞춰야 한다. 포켓몬의 시각적 특징과 속성을 모두 고려하여 단계별로 추리하고, 가장 가능성이 높은 최종 후보 포켓몬 한 마리의 이름만 정확하게 말해줘.\n\n[힌트]\n{full_text}\n\n[최종 결론 (포켓몬 이름만)]"
            
            response = client.chat.completions.create(
                model="gpt-4o", 
                messages=[{"role": "user", "content": prompt}]
            )
            predicted_name = response.choices[0].message.content.strip().split('\n')[-1]
            
            result_df = df[df['name_ko'].str.contains(predicted_name, case=False)]
            
            if not result_df.empty:
                best_pokemon = result_df.iloc[0]
                st.success("이 포켓몬이 아닐까요?")
                st.image(best_pokemon['image_url'], width=250)
                st.write(f"### {best_pokemon['name_ko']}")
            else:
                st.warning(f"'{predicted_name}'... 으음, 아직은 잘 모르겠군요. 힌트를 더 주세요!")

def draw_chatbot_view():
    st.header("오박사 AI 챗봇")
    
    if client is None: 
        st.warning("OpenAI API 키를 설정하면 이 기능을 사용할 수 있습니다.")
        return
    
    if 'chat_history' not in st.session_state: 
        st.session_state.chat_history = []
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    audio = mic_recorder(start_prompt="🎤 질문하기", stop_prompt="⏹️ 녹음중지", just_once=True, key='chatbot_mic')
    
    if audio and audio.get('bytes'):
        with st.spinner("음성을 텍스트로 변환 중..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio['bytes'])
                file_path = f.name
            
            with open(file_path, "rb") as audio_file:
                user_text = client.audio.transcriptions.create(model="whisper-1", file=audio_file).text
            
            os.remove(file_path)
        
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        
        with st.chat_message("user"):
            st.write(user_text)
        
        with st.spinner("오박사님이 답변을 생각하는 중..."):
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                system_prompt = "너는 포켓몬스터에 나오는 오박사야. 포켓몬을 굉장히 잘 알아. 친절하고 상세하게 설명해줘."
                
                messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
                stream = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api, stream=True)
                
                for chunk in stream:
                    full_response += (chunk.choices[0].delta.content or "")
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

# 전역 실행 부분 - 앱 시작 시 자동으로 실행됩니다
add_custom_css()
st.markdown("<h1 class='main-title'>P.R.I.S.M.</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px; color: #00aaff; font-family: Orbitron;'>Pokemon Research Intelligence System Matrix</p>", unsafe_allow_html=True)
st.sidebar.link_button("🔴 포켓몬 RPG 세계로 떠나기🔴", "http://localhost:8502")

# 모델과 데이터 로드
model, preprocess = load_clip_model()
df = fetch_and_process_pokemon_data()
embeddings = create_clip_embeddings(df, model, preprocess)

# 사이드바 구성
filtered_df = draw_sidebar_one(df) 
draw_sidebar_two(df)

# 메인 탭 구성
tabs = st.tabs([
    "[ 도감 탐색기 ]", 
    "[ 시냅스 공명 분석기 ]", 
    "[ 포켓몬 유추 게임 ]", 
    "[ 스무고개 AI ]", 
    "[ 오박사 AI 챗봇 ]"
])

with tabs[0]:
    draw_pokedex_view(filtered_df)

with tabs[1]:
    draw_resonance_analyzer_view(df, embeddings)

with tabs[2]:
    draw_analogy_game_view(df, embeddings)

with tabs[3]:
    draw_20q_game_view(df)

with tabs[4]:
    draw_chatbot_view()