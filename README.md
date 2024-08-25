[2. Bibliotecas Python:](https://github.com/bombomdesal/teste/blob/main/Bibliotecas%20Python/Vis%C3%A3o%20geral.md)
   
[Pandas](https://github.com/bombomdesal/teste/blob/main/Bibliotecas%20Python/Pandas.md) (manipulação;limpeza; transformação e pré-processamento de dado);

[NumPy](https://github.com/bombomdesal/teste/blob/main/Bibliotecas%20Python/Numpy.md) (operações de arrays);

[Matplotlib](https://github.com/bombomdesal/teste/blob/main/Bibliotecas%20Python/Matplotlib.md) e [Seaborn](https://github.com/bombomdesal/teste/blob/main/Bibliotecas%20Python/Seaborn.md) (visualização de dados); 

TensorFlow; Keras e PyTorch (redes neurais); 

Scikit-learn e XGBoost (aprendizado de máquina); 

NLTK e spaCy (processamento de linguagem natural); 

huggingface (LLM); 

PySpark (Big data); 

Beautiful Soup (web scraping); 

Streamlit (data apps)


[português](https://github.com/bombomdesal/teste/blob/main/portugues/ortografia.md)

[redes neurais](https://github.com/bombomdesal/teste/blob/main/modelagem/redes_neurais.md)


Aqui estão os passos para criar um repositório Git no Termux, adicionar pastas e arquivos Markdown, e conectá-lo a um repositório no GitHub:

1. **Navegar para o diretório desejado no Termux:**
   ```bash
   cd /caminho/do/diretório
   ```

2. **Criar o repositório Git:**
   ```bash
   git init
   ```

3. **Criar pastas e arquivos Markdown:**
   Para criar pastas:
   ```bash
   mkdir pasta1 pasta2
   ```
   Para criar arquivos Markdown dentro dessas pastas:
   ```bash
   echo "# Título do Markdown" > pasta1/arquivo1.md
   echo "# Outro Markdown" > pasta2/arquivo2.md
   ```

4. **Adicionar os arquivos ao repositório:**
   ```bash
   git add .
   ```

5. **Fazer o primeiro commit:**
   ```bash
   git commit -m "Primeiro commit"
   ```

6. **Criar um repositório no GitHub:**
   - Vá ao [GitHub](https://github.com) e crie um novo repositório.
   - Copie o link do repositório remoto (algo como `https://github.com/username/repo.git`).

7. **Conectar o repositório local ao GitHub:**
   ```bash
   git remote add origin https://github.com/username/repo.git
   ```

8. **Enviar o commit para o GitHub:**
   ```bash
   git push -u origin master
   ```

Após seguir esses passos, seu repositório no Termux estará conectado ao GitHub e os arquivos Markdown estarão disponíveis no repositório remoto.
