# 🌀 Sonic Mania - Detecção de Estados dos Death Eggs com OpenCV

Este projeto utiliza **visão computacional (OpenCV)** para identificar e rastrear os **estados de ataque dos Death Eggs** no jogo *Sonic Mania*.  
A ideia central é capturar frames do jogo, processá-los em uma **grade (grid)** e aplicar técnicas de **detecção de cores e padrões** para entender o comportamento dos inimigos durante a batalha.

---

## 🎮 Contexto

No jogo *Sonic Mania*, os **Death Eggs** possuem diferentes estados visuais durante o ataque — variando em cor, brilho e movimento.  
Essas variações podem indicar:
- **Modo normal**  
- **Modo de ataque**  
- **Modo danificado**  
- **Modo crítico**

Através do uso de **OpenCV**, o projeto tenta reconhecer automaticamente essas mudanças analisando a distribuição de pixels vermelhos, amarelos e brancos em uma **malha (grid)** sobre a tela.

---

## 🧠 Objetivo

Desenvolver um modelo simples e interpretável capaz de:
1. **Detectar** a presença dos Death Eggs na imagem;
2. **Analisar o padrão de cor e movimento** em regiões específicas (grid);
3. **Classificar o estado atual** (ex: ataque, neutro, dano);
4. **Fornecer uma visualização** do rastreamento em tempo real.

---

## ⚙️ Tecnologias Utilizadas

- 🐍 **Python 3.x**
- 👁️ **OpenCV** (`cv2`)
- 🔢 **NumPy**
- 🧮 **scikit-learn** (para normalização e classificação)
- 💾 **pickle** (para salvar e carregar modelos)
- 🧱 **StandardScaler** (pré-processamento)
- 🤖 **MLPClassifier** (classificador de estados com rede neural)

---

## 🧩 Estrutura do Projeto

