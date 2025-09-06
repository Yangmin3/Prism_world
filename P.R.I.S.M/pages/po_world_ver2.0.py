# po_world_final.py
import streamlit as st
import json
import os
import random
import base64
import requests
from openai import OpenAI
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv

# --- 1. 초기 설정 ---
load_dotenv()
# !!! 중요 !!! 아래 "YOUR_API_KEY" 부분에 본인의 실제 OpenAI API 키를 입력하세요.
api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
client = OpenAI(api_key=api_key) if "YOUR_API_KEY" not in api_key and api_key else None

st.set_page_config(page_title="Pokémon RPG", page_icon="🔴", layout="centered")

# --- 2. 사운드, UI/UX, 게임 데이터 ---
SFX_SELECT_B64 = "data:audio/wav;base64,UklGRiQSAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQASAADp/uL+5f7m/uf+6P7p/un+6v7q/ur+7P7s/uz+7f7t/u3+7v7u/u7++f75/vn+AAMA//76/vn+9v72/vb+9f71/vX+8/7z/vP+8P7w/vD+7v7u/u4="
SFX_HIT_B64 = "data:audio/wav;base64,UklGRjwPAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQCEDwAAPj7e/uL+Zf7g/lX+5/6C/sP+ov7w/kf/ev+C/4r/1P/3//X/AwMLAi0GDBQxGicwMjYpLi8wGyQgFRUXFBAQEA8NDAsLCwoKCQUFBQUDAQABAQEAAA=="
DEFAULT_POKEBALL_IMAGE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/poke-ball.png"

TYPE_CHART = {
    "노말": {"ineffective": ["바위", "강철"], "immune": ["고스트"]},
    "불꽃": {"effective": ["풀", "얼음", "벌레", "강철"], "ineffective": ["불꽃", "물", "바위", "드래곤"]},
    "물": {"effective": ["불꽃", "땅", "바위"], "ineffective": ["물", "풀", "드래곤"]},
    "풀": {"effective": ["물", "땅", "바위"], "ineffective": ["불꽃", "풀", "독", "비행", "벌레", "드래곤", "강철"]},
    "전기": {"effective": ["물", "비행"], "ineffective": ["전기", "풀", "드래곤"], "immune": ["땅"]},
    "격투": {"effective": ["노말", "얼음", "바위", "악", "강철"], "ineffective": ["독", "비행", "에스퍼", "벌레", "페어리"], "immune": ["고스트"]},
    "비행": {"effective": ["풀", "격투", "벌레"], "ineffective": ["전기", "바위", "강철"]},
    "독": {"effective": ["풀", "페어리"], "ineffective": ["독", "땅", "바위", "고스트"], "immune": ["강철"]},
}
NPCS = {
    "반바지소년 영민": {
        "party": [{ "id": 19, "name": "꼬렛", "level": 5, "attack": 9, "hp": 20, "max_hp": 20, "type": ["노말"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/19.png"}]
    }
}

def add_custom_css():
    bg_image_path = "meadow_background.png"
    if os.path.exists(bg_image_path):
        with open(bg_image_path, "rb") as f:
            bg_image = base64.b64encode(f.read()).decode()
    else:
        st.warning("배경 이미지 파일(meadow_background.png)이 없어 기본 배경으로 실행됩니다.")
        bg_image = ""
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DotGothic16&display=swap');
            .stApp {{ background-image: url("data:image/png;base64,{bg_image}"); background-size: cover; }}
            * {{ font-family: 'DotGothic16', sans-serif; }}
            .game-screen {{ background-color: rgba(10, 20, 40, 0.9); border: 4px solid #4a7d9f; border-radius: 15px; padding: 20px; margin-top: 20px; }}
            .game-screen p {{ color: #ffffff !important; font-size: 1.2em; }}
            .hp-bar-container {{ background-color: #555; border-radius: 5px; padding: 2px; border: 1px solid #333; }}
            .hp-bar {{ height: 10px; border-radius: 3px; transition: width 0.5s ease-in-out; }}
            .hp-high {{ background: linear-gradient(to right, #58e572, #3baa4c); }}
            .hp-mid {{ background: linear-gradient(to right, #f7d354, #daa520); }}
            .hp-low {{ background: linear-gradient(to right, #f75454, #aa3b3b); }}
            .exp-bar-container {{ background-color: #444; border-radius: 5px; padding: 1px; border: 1px solid #222; margin-top: -15px;}}
            .exp-bar {{ background: linear-gradient(to right, #54b2f7, #2c89d8); height: 5px; border-radius: 3px; }}
            .battle-container {{ position: relative; height: 350px; }}
            .enemy-pokemon {{ position: absolute; top: 20px; right: 20px; text-align: center; border: 2px solid #fff; border-radius: 10px; background-color: rgba(0,0,0,0.5); padding: 5px; }}
            .player-pokemon {{ position: absolute; bottom: 80px; left: 20px; text-align: center; border: 2px solid #fff; border-radius: 10px; background-color: rgba(0,0,0,0.5); padding: 5px; }}
        </style>
    """, unsafe_allow_html=True)

REGISTERED_USERS = {"yang": {"password": "mm", "nickname": "정민"}, "han": {"password": "HH", "nickname": "후니부기"}}
STARTER_POKEMON = {
    1: [{"name": "이상해씨", "id": 1, "type": ["풀", "독"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/1.png"},
        {"name": "파이리", "id": 4, "type": ["불꽃"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/4.png"},
        {"name": "꼬부기", "id": 7, "type": ["물"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/7.png"}],
    2: [{"name": "치코리타", "id": 152, "type": ["풀"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/152.png"},
        {"name": "브케인", "id": 155, "type": ["불꽃"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/155.png"},
        {"name": "리아코", "id": 158, "type": ["물"], "sprite": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/158.png"}],
}

# --- 4. 보조 함수 ---
def get_type_effectiveness(attack_type, defense_types):
    effectiveness = 1.0
    if attack_type in TYPE_CHART:
        for def_type in defense_types:
            if def_type in TYPE_CHART[attack_type].get("effective", []): effectiveness *= 2.0
            if def_type in TYPE_CHART[attack_type].get("ineffective", []): effectiveness *= 0.5
            if def_type in TYPE_CHART[attack_type].get("immune", []): effectiveness = 0.0; break
    return effectiveness

def play_sfx(sfx_b64):
    st.markdown(f'<audio controls autoplay="true" style="display:none;"><source src="{sfx_b64}" type="audio/wav"></audio>', unsafe_allow_html=True)

def play_tts(text):
    if client is None: return
    try:
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        b64 = base64.b64encode(response.content).decode()
        md = f'<audio controls autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"TTS 실행 중 오류 발생: {e}")

def load_user_data(username):
    filepath = f"user_data_{username}.json"
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f: return json.load(f)
    return None

def save_user_data(username, data):
    filepath = f"user_data_{username}.json"
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=4)

@st.cache_data
def get_evolution_info(pokemon_id):
    try:
        species_res = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{pokemon_id}/").json()
        evolution_chain_url = species_res['evolution_chain']['url']
        chain_res = requests.get(evolution_chain_url).json()
        chain = chain_res['chain']
        
        current_pokemon_name = species_res['name']
        while chain:
            if chain['species']['name'] == current_pokemon_name and chain['evolves_to']:
                evolution_details = chain['evolves_to'][0]['evolution_details'][0]
                if evolution_details['trigger']['name'] == 'level-up' and evolution_details['min_level']:
                    next_pokemon_name_en = chain['evolves_to'][0]['species']['name']
                    next_pokemon_res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{next_pokemon_name_en}/").json()
                    ko_name_obj = next((n for n in next_pokemon_res['species']['names'] if n['language']['name'] == 'ko'), None)
                    ko_name = ko_name_obj['name'] if ko_name_obj else next_pokemon_name_en
                    return {"next_form_id": next_pokemon_res['id'], "next_form_name": ko_name, "min_level": evolution_details['min_level']}
            if not chain['evolves_to']: break
            chain = chain['evolves_to'][0]
        return None
    except Exception:
        return None

@st.cache_data
def get_official_moves(pokemon_id, level):
    try:
        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}").json()
        learnable_moves = []
        for move_data in res['moves']:
            for version_detail in move_data['version_group_details']:
                if (version_detail['move_learn_method']['name'] == 'level-up' and version_detail['level_learned_at'] <= level):
                    move_name_res = requests.get(move_data['move']['url']).json()
                    ko_move_name = next((name['name'] for name in move_name_res['names'] if name['language']['name'] == 'ko'), move_data['move']['name'])
                    learnable_moves.append((version_detail['level_learned_at'], ko_move_name))
        
        learnable_moves.sort(key=lambda x: (x[0], x[1]), reverse=True)
        final_moves = list(dict.fromkeys([move[1] for move in learnable_moves]))[:4]
        return final_moves if final_moves else ["몸통박치기"]
    except Exception:
        return ["몸통박치기", "울음소리"]

def call_llm_gm(player_data, action, battle_result=None):
    recent_logs = "\n".join(player_data.get("game_log", [])[-3:])
    battle_context = ""
    if battle_result:
        battle_context = f"[방금 일어난 이벤트 결과 요약]\n{json.dumps(battle_result, ensure_ascii=False, indent=2)}"

    prompt = f"""
    당신은 포켓몬 펄/디아루가 스타일의 텍스트 RPG 게임 마스터(GM)입니다.
    {battle_context}
    [최근 이벤트 요약]\n{recent_logs}
    [현재 플레이어 상태]\n{json.dumps(player_data, ensure_ascii=False, indent=2)}
    [플레이어 행동]\n{action}
    [규칙]
    - `narration`에는 상황을 흥미진진하고 간결하게 묘사해주세요.
    - 전투가 끝났다면(`enemy`의 hp가 0 이하), 전투 종료를 선언하고 `exp_gained` 필드에 획득 경험치를 숫자로 반환해주세요.
    - `battle_status`는 `player_turn`, `enemy_turn`, `inactive` 중 하나여야 합니다.
    - `enemy` 등장 시, 모든 정보를 포함해야 합니다.
    - 반드시 지정된 JSON 구조로만 반환해야 합니다.
    {{
        "narration": "묘사...", "exp_gained": 50, "player_data": {{...}}, "options": ["행동1"], "battle_status": "inactive",
        "enemy": {{ "name": "구구", "id": 16, "level": 3, "hp": 15, "max_hp": 15, "attack": 8, "sprite": "...", "type": ["노말", "비행"] }}
    }}
    """
    if not client: return None
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", response_format={"type": "json_object"}, messages=[{"role": "system", "content": "You are a game master. Respond in valid JSON."}, {"role": "user", "content": prompt}])
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"LLM 호출 오류: {e}")
        return None

def trigger_random_event():
    events = [
        {"type": "item", "name": "상처약", "message": "풀숲을 뒤지자 반짝이는 무언가를 발견했다! **상처약**을 1개 손에 넣었다!"},
        {"type": "heal", "message": "상냥해 보이는 할머니를 만났다. 할머니는 당신의 포켓몬들을 모두 치료해주었다!"},
        {"type": "npc_battle", "npc_name": "반바지소년 영민", "message": "저쪽에서 한 소년이 달려오더니 눈이 마주쳤다! '나와 포켓몬 승부다!'"},
    ]
    return random.choice(events)

def handle_simple_event(event):
    player_data = st.session_state.player_data
    message = event['message']
    if event['type'] == 'item':
        item_found = next((item for item in player_data['items'] if item['name'] == event['name']), None)
        if item_found:
            item_found['quantity'] += 1
        else:
            player_data['items'].append({'name': event['name'], 'quantity': 1})
    elif event['type'] == 'heal':
        for p in player_data['pokemon_party']:
            p['hp'] = p['max_hp']
    return message, player_data

# --- 5. UI 그리기 함수 ---
def draw_login():
    st.title("🔴 포켓몬 RPG 🔴")
    with st.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("모험 시작")
        if submitted:
            play_sfx(SFX_SELECT_B64)
            if username in REGISTERED_USERS and REGISTERED_USERS[username]["password"] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.nickname = REGISTERED_USERS[username]["nickname"]
                st.session_state.player_data = load_user_data(username)
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")

def draw_starter_selection():
    st.header(f"{st.session_state.nickname}님, 파트너를 선택하세요!")
    if "tts_played" not in st.session_state:
        # play_tts("포켓몬 세계에 잘 왔다. 너와 함께 할 포켓몬을 고르도록.")
        st.session_state.tts_played = True
    
    gen_tabs = st.tabs([f"{i}세대" for i in STARTER_POKEMON.keys()])
    for i, tab in enumerate(gen_tabs):
        gen = i + 1
        with tab:
            cols = st.columns(3)
            for j, starter in enumerate(STARTER_POKEMON[gen]):
                with cols[j]:
                    st.image(starter["sprite"], width=150)
                    if st.button(f"{starter['name']} 선택", key=f"select_{gen}_{j}"):
                        play_sfx(SFX_SELECT_B64)
                        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{starter['id']}").json()
                        back_sprite = res['sprites']['back_default']
                        initial_moves = get_official_moves(starter['id'], 5)
                        st.session_state.player_data = {
                            "location": "1번 도로", "gold": 500,
                            "pokemon_party": [{"id": starter["id"], "name": starter["name"], "type": starter["type"], "level": 5, "hp": 25, "max_hp": 25, "attack": 10, "exp": 0, "exp_to_next_level": 50, "sprite": starter["sprite"], "back_sprite": back_sprite, "moves": initial_moves}],
                            "items": [{"name": "몬스터볼", "quantity": 5}, {"name": "상처약", "quantity": 3}],
                        }
                        start_message = f"{starter['name']}와(과) 함께 당신의 모험이 시작되었다!"
                        st.session_state.player_data["game_log"] = [start_message]
                        st.session_state.game_state = {"narration": start_message, "battle_status": "inactive"}
                        save_user_data(st.session_state.username, st.session_state.player_data)
                        st.rerun()

def draw_hp_bar(current_hp, max_hp):
    percent = (current_hp / max_hp) * 100
    if percent >= 70: color_class = "hp-high"
    elif percent >= 30: color_class = "hp-mid"
    else: color_class = "hp-low"
    st.markdown(f'<div class="hp-bar-container"><div class="hp-bar {color_class}" style="width: {percent}%;"></div></div><p style="text-align: right; margin-top: -5px; font-size: 0.9em;">{current_hp} / {max_hp}</p>', unsafe_allow_html=True)

def draw_exp_bar(current_exp, exp_to_next):
    percent = min(100.0, (current_exp / exp_to_next) * 100 if exp_to_next > 0 else 100)
    st.markdown(f'<div class="exp-bar-container"><div class="exp-bar" style="width: {percent}%;"></div></div>', unsafe_allow_html=True)

def draw_main_game_ui():
    player_data = st.session_state.player_data
    game_state = st.session_state.get("game_state", {})
    st.sidebar.header(f"{st.session_state.nickname}의 정보")
    st.sidebar.write(f"💰 골드: {player_data.get('gold', 0)} G")
    
    st.sidebar.subheader("포켓몬 파티")
    for p in player_data["pokemon_party"]:
        st.sidebar.image(p.get("sprite", DEFAULT_POKEBALL_IMAGE))
        st.sidebar.write(f"**{p['name']}** (Lv. {p['level']})")
        draw_hp_bar(p["hp"], p["max_hp"])
        draw_exp_bar(p["exp"], p["exp_to_next_level"])
    
    st.sidebar.subheader("가방")
    for item in player_data["items"]:
        st.sidebar.write(f"- {item['name']} ({item['quantity']}개)")
    
    st.title(f"현재 위치: {player_data['location']}")

    with st.container():
        st.markdown('<div class="game-screen">', unsafe_allow_html=True)
        
        if game_state.get("battle_status", "inactive") != "inactive" and game_state.get("enemy"):
            enemy = game_state["enemy"]
            player_pokemon = player_data["pokemon_party"][0]
            st.markdown('<div class="battle-container">', unsafe_allow_html=True)
            with st.container():
                st.markdown(f'<div class="enemy-pokemon">', unsafe_allow_html=True)
                st.image(enemy.get("sprite") or DEFAULT_POKEBALL_IMAGE, width=150)
                st.markdown(f'<p>{enemy.get("name", "???")} Lv.{enemy.get("level", "?")}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="player-pokemon">', unsafe_allow_html=True)
                st.image(player_pokemon.get("back_sprite") or player_pokemon.get("sprite") or DEFAULT_POKEBALL_IMAGE, width=200)
                st.markdown(f'<p>{player_pokemon["name"]} Lv.{player_pokemon["level"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        narration = game_state.get("narration", player_data.get("game_log", ["모험을 시작했다."])[-1])
        st.markdown(f"<p>{narration}</p>", unsafe_allow_html=True)

        if st.button("🔊 음성으로 듣기", key="tts_button"):
            play_tts(narration)
        
        st.markdown("---")

        if game_state.get("status") == "awaiting_evolution":
            pokemon_to_evolve = player_data["pokemon_party"][0]
            st.subheader(f"어라? {pokemon_to_evolve['name']}의 상태가...!")
            cols = st.columns(2)
            if cols[0].button("진화시킨다", use_container_width=True):
                handle_action("진화시킨다", game_state)
            if cols[1].button("그만둔다", use_container_width=True):
                handle_action("진화를 멈춘다", game_state)
        
        elif game_state.get("status") == "learning_move":
            new_move = game_state["new_move"]
            pokemon = player_data["pokemon_party"][0]
            st.info(f"{pokemon['name']}은(는) 새로운 기술 **{new_move}**을(를) 배우려 한다! 배우게 할까?")
            col1, col2 = st.columns(2)
            if col1.button(f"{new_move}을(를) 배운다", use_container_width=True):
                if len(pokemon["moves"]) < 4:
                    handle_action(f"기술 '{new_move}' 배우기", game_state)
                else:
                    st.session_state.game_state["status"] = "replacing_move"
                    st.rerun()
            if col2.button("배우지 않는다", use_container_width=True):
                handle_action("기술 배우기 취소", game_state)

        elif game_state.get("status") == "replacing_move":
            new_move = game_state["new_move"]
            pokemon = player_data["pokemon_party"][0]
            st.warning(f"기술은 4개까지밖에 배울 수 없다! 어떤 기술을 잊게 할까?")
            move_to_forget = st.radio("잊을 기술을 선택하세요:", pokemon["moves"], key="forget_move_radio")
            if st.button(f"{move_to_forget}을(를) 잊는다", use_container_width=True):
                handle_action(f"'{move_to_forget}'을(를) 잊고 '{new_move}' 배우기", game_state)

        elif game_state.get("battle_status") == "player_turn":
            if 'battle_action' not in st.session_state: st.session_state.battle_action = None
            battle_options = st.columns(4)
            if battle_options[0].button("싸운다"): play_sfx(SFX_SELECT_B64); st.session_state.battle_action = "fight"; st.rerun()
            if battle_options[1].button("가방"): play_sfx(SFX_SELECT_B64); st.session_state.battle_action = "bag"; st.rerun()
            if battle_options[2].button("포켓몬"): play_sfx(SFX_SELECT_B64); st.warning("포켓몬 교체는 다음 버전에서 만나요!")
            if battle_options[3].button("도망간다"): play_sfx(SFX_SELECT_B64); handle_action("전투에서 도망친다", game_state)

            if st.session_state.battle_action == "fight":
                st.subheader("어떤 기술을 사용할까?")
                moves = player_pokemon.get("moves", [])
                if moves:
                    move_cols = st.columns(len(moves))
                    for i, move in enumerate(moves):
                        if move_cols[i].button(move, key=f"move_{move}"):
                            handle_action(f"기술 '{move}' 사용", game_state)
                else:
                    st.warning("사용할 수 있는 기술이 없습니다!")
                if st.button("뒤로가기"):
                    st.session_state.battle_action = None
                    st.rerun()

            elif st.session_state.battle_action == "bag":
                st.subheader("어떤 아이템을 사용할까?")
                for item in player_data.get("items", []):
                    if st.button(f"{item['name']} ({item['quantity']}개)", key=item['name']):
                        handle_action(f"아이템 '{item['name']}' 사용", game_state)
                if st.button("가방 닫기"):
                    st.session_state.battle_action = None
                    st.rerun()

        elif game_state.get("battle_status") == "enemy_turn":
            if st.button("계속...", use_container_width=True):
                handle_action("적의 턴 진행", game_state)

        else: # 평상시
            options = game_state.get("options", ["풀숲을 탐색한다", "마을을 둘러본다"])
            if random.random() < 0.2:
                options.append("✨ 주변을 자세히 살피기")
            unique_options = list(dict.fromkeys(options))
            cols = st.columns(len(unique_options))
            for i, option in enumerate(unique_options):
                if cols[i].button(option, key=option):
                    play_sfx(SFX_SELECT_B64)
                    if option == "✨ 주변을 자세히 살피기":
                        event = trigger_random_event()
                        if event["type"] == "npc_battle":
                            npc = NPCS[event["npc_name"]]
                            enemy_pokemon_data = npc["party"][0]
                            enemy_pokemon_data["moves"] = get_official_moves(enemy_pokemon_data["id"], enemy_pokemon_data["level"])
                            st.session_state.game_state = {"narration": event["message"], "battle_status": "player_turn", "enemy": enemy_pokemon_data, "is_npc_battle": True}
                            st.rerun()
                        else:
                            message, player_data = handle_simple_event(event)
                            st.session_state.player_data = player_data
                            st.session_state.game_state['narration'] = message
                            st.session_state.player_data.setdefault("game_log", []).append(message)
                            save_user_data(st.session_state.username, player_data)
                            st.rerun()
                    else:
                        handle_action(option, game_state)
        st.markdown('</div>', unsafe_allow_html=True)

def handle_action(action, game_state):
    player_data = st.session_state.player_data
    battle_result = None

    if "기술" in action and "사용" in action:
        player_pokemon = player_data["pokemon_party"][0]
        enemy_pokemon = game_state["enemy"]
        if random.random() < 0.1:
            damage = 0
            effectiveness_text = "공격이 빗나갔다!"
        else:
            damage = player_pokemon["attack"] + random.randint(-2, 2)
            effectiveness = get_type_effectiveness(player_pokemon["type"][0], enemy_pokemon["type"])
            damage = int(damage * effectiveness)
            if effectiveness > 1: effectiveness_text = "효과는 굉장했다!"
            elif effectiveness < 1 and effectiveness > 0: effectiveness_text = "효과가 별로인 듯하다..."
            elif effectiveness == 0: effectiveness_text = "효과가 없는 것 같다..."
            else: effectiveness_text = ""
            if random.random() < 0.125:
                damage = int(damage * 1.5)
                effectiveness_text += " 급소에 맞았다!"
        battle_result = {"action_type": "attack", "player_move": action, "damage_dealt": damage, "effectiveness": effectiveness_text}

    elif "아이템" in action and "사용" in action:
        item_name = action.replace("아이템 '", "").replace("' 사용", "")
        item_found = next((item for item in player_data["items"] if item["name"] == item_name), None)
        if item_found and item_found["quantity"] > 0:
            if item_name == "몬스터볼":
                enemy = game_state["enemy"]
                catch_chance = max(0.1, (1 - (enemy['hp'] / enemy['max_hp'])) * 0.85)
                if random.random() < catch_chance:
                    battle_result = {"action_type": "catch", "result": "success"}
                    new_pokemon = {**enemy}
                    new_pokemon['moves'] = get_official_moves(new_pokemon.get('id'), new_pokemon.get('level'))
                    if len(player_data["pokemon_party"]) < 6:
                        player_data["pokemon_party"].append(new_pokemon)
                    else:
                        battle_result = {"action_type": "catch", "result": "party_full"}
                else:
                    battle_result = {"action_type": "catch", "result": "failure"}
            elif item_name == "상처약":
                player_pokemon = player_data["pokemon_party"][0]
                player_pokemon["hp"] = min(player_pokemon["max_hp"], player_pokemon["hp"] + 20)
                battle_result = {"action_type": "item_use", "result": f"{player_pokemon['name']}의 HP를 20 회복했다!"}
            item_found["quantity"] -= 1
            if item_found["quantity"] == 0:
                player_data["items"].remove(item_found)
        else:
            battle_result = {"action_type": "item_use", "result": "아이템이 없어 사용할 수 없다!"}

    elif action == "진화시킨다":
        pokemon = player_data["pokemon_party"][0]
        evo_info = game_state["evolution_info"]
        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{evo_info['next_form_id']}").json()
        original_name = pokemon['name']
        pokemon['name'] = evo_info['next_form_name']
        pokemon['id'] = res['id']
        pokemon['sprite'] = res['sprites']['front_default']
        pokemon['back_sprite'] = res['sprites']['back_default']
        types_res = [type_data['type']['name'] for type_data in res['types']]
        ko_types = []
        for t in types_res:
            type_res = requests.get(f"https://pokeapi.co/api/v2/type/{t}").json()
            ko_types.append(next((n['name'] for n in type_res['names'] if n['language']['name'] == 'ko'), t))
        pokemon['type'] = ko_types
        pokemon['max_hp'] += 10; pokemon['hp'] = pokemon['max_hp']; pokemon['attack'] += 5
        st.session_state.game_state = {"narration": f"축하합니다! 당신의 {original_name}은(는) {pokemon['name']}(으)로 진화했다!", "status": "active"}
        st.rerun(); return
    
    elif action == "진화를 멈춘다":
        st.session_state.game_state = {"narration": f"{player_data['pokemon_party'][0]['name']}의 몸에서 빛이 사라졌다...", "status": "active"}
        st.rerun(); return
    
    elif "배우기" in action:
        pokemon = player_data["pokemon_party"][0]
        new_move = game_state["new_move"]
        if "취소" in action:
            st.session_state.game_state = {"narration": f"{pokemon['name']}은(는) 기술을 배우지 않았다.", "status": "active"}
        elif "잊고" in action:
            move_to_forget = action.split("'")[1]
            move_index = pokemon["moves"].index(move_to_forget)
            pokemon["moves"][move_index] = new_move
            st.session_state.game_state = {"narration": f"{pokemon['name']}은(는) {move_to_forget}을(를) 잊고, **{new_move}**을(를) 배웠다!", "status": "active"}
        else:
            pokemon["moves"].append(new_move)
            st.session_state.game_state = {"narration": f"{pokemon['name']}은(는) {new_move}을(를) 배웠다!", "status": "active"}
        st.rerun(); return

    new_state = call_llm_gm(player_data, action, battle_result)
    
    if new_state:
        if game_state.get("battle_status") != "active" and new_state.get("battle_status") == "active":
            play_sfx(SFX_HIT_B64)
            if "enemy" in new_state and new_state["enemy"]:
                enemy_id = new_state["enemy"].get("id")
                enemy_level = new_state["enemy"].get("level")
                if enemy_id and enemy_level:
                    new_state["enemy"]["moves"] = get_official_moves(enemy_id, enemy_level)

        if game_state.get("battle_status") != "inactive" and new_state.get("battle_status") == "inactive":
            exp_gained = new_state.get("exp_gained", 0)
            if exp_gained > 0:
                pokemon = new_state["player_data"]["pokemon_party"][0]
                if pokemon["level"] < 100:
                    pokemon["exp"] += exp_gained
                    new_state["narration"] += f"\n\n{pokemon['name']}은(는) 경험치 {exp_gained}을(를) 얻었다!"
                    while pokemon["exp"] >= pokemon["exp_to_next_level"] and pokemon["level"] < 100:
                        pokemon["level"] += 1
                        pokemon["exp"] -= pokemon["exp_to_next_level"]
                        pokemon["exp_to_next_level"] = int(pokemon["exp_to_next_level"] * 1.5)
                        pokemon["max_hp"] += 5; pokemon["attack"] += 2; pokemon["hp"] = pokemon["max_hp"]
                        
                        evolution_info = get_evolution_info(pokemon["id"])
                        if evolution_info and pokemon["level"] >= evolution_info["min_level"]:
                            new_state["status"] = "awaiting_evolution"; new_state["evolution_info"] = evolution_info; break
                        else:
                            new_move_list = get_official_moves(pokemon["id"], pokemon["level"])
                            if new_move_list and new_move_list[0] not in updated_pokemon["moves"]:
                                new_state["status"] = "learning_move"; new_state["new_move"] = new_move_list[0]; break
            
            if game_state.get("is_npc_battle"):
                gold_won = game_state.get("enemy", {}).get("level", 5) * 20
                new_state["player_data"]["gold"] = player_data.get("gold", 0) + gold_won
                new_state["narration"] += f"\n\n상대에게서 용돈 {gold_won} G를 받았다!"
        
        for p in new_state.get("player_data", {}).get("pokemon_party", []):
            if "back_sprite" not in p or p["back_sprite"] is None:
                try:
                    if "id" not in p: p["id"] = p["sprite"].split('/')[-1].split('.')[0]
                    res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{p['id']}").json()
                    p["back_sprite"] = res['sprites']['back_default']
                    if p["back_sprite"] is None: p["back_sprite"] = p["sprite"]
                except Exception:
                    p["back_sprite"] = p.get("sprite", DEFAULT_POKEBALL_IMAGE)

        st.session_state.game_state = new_state
        st.session_state.player_data = new_state["player_data"]
        st.session_state.player_data.setdefault("game_log", []).append(new_state.get("narration", ""))
        save_user_data(st.session_state.username, new_state["player_data"])
        st.rerun()

# --- 6. 메인 실행 로직 ---
add_custom_css()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    draw_login()
else:
    if st.session_state.get("player_data") is None:
        draw_starter_selection()
    elif client:
        draw_main_game_ui()
    else:
        st.error("OpenAI API 키가 설정되지 않아 게임을 진행할 수 없습니다.")