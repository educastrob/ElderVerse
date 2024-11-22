# ElderVerse Chatbot

ElderVerse Chatbot é um chatbot amigável e paciente projetado para engajar em conversas com indivíduos idosos. Ele demonstra um interesse genuíno em suas histórias e experiências, e gera uma postagem de blog atraente a partir da conversa, salvando-a como um arquivo PDF.

## Funcionalidades

- Engaja em conversas baseadas em texto com indivíduos idosos.
- Gera uma postagem de blog atraente a partir da conversa.
- Salva a postagem do blog como um arquivo PDF.

## Requisitos

- Python 3.x

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/Vinni-Cedraz/ElderVerse.git
   cd ElderVerse
   ```

2. Defina a variável de ambiente `GROQ_API_KEY`:
   ```bash
   export GROQ_API_KEY=sua_chave_groq_api_key
   ```

3. Crie o ambiente virtual:
   ```bash
   python -m venv myenv
   ```
4. Ative o ambiente virtual:
    ```bash
    source myenv/bin/activate
    ```
5. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

Execute o chatbot:
```bash
./main.py
```

O chatbot iniciará uma conversa e gerará uma postagem de blog a partir da conversa como um PDF quando você digitar `quit`.

## Licença

Este projeto está licenciado sob a Licença MIT.
