import streamlit as st

from llm import get_ai_message


st.set_page_config(page_title="소득세 도우미", page_icon="💵")
st.title("💵 소득세 도우미")
st.caption("소득세법에 대해서 알려드립니다!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_question := st.chat_input(
    placeholder="소득세법에 대해서 궁금한 점을 물어봐주세요!!"
):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("조금만 기다려주세요🥹"):
        ai_message = get_ai_message(user_question)

        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.messages.append({"role": "ai", "content": ai_message})
