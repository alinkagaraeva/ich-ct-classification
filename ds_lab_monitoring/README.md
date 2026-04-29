# Data Secrets Lab

![Кот](TypingCat.png)

Добро пожаловать в **Data Secrets Lab**! 🎯

---

## 🐍 Python окружение

В этом контейнере используется **глобальная установка Python** без виртуальных окружений (venv).

### Почему нет venv?
- **Один проект = один контейнер** — изоляция уже обеспечена Docker
- Не нужно активировать окружение — всё работает сразу
- Меньше путаницы с путями и интерпретаторами

### Установка пакетов
Просто используйте pip:
```bash
pip install numpy pandas matplotlib
pip install torch torchvision torchaudio
pip install transformers datasets
```

---

## 🖥️ Что под капотом

| Компонент | Версия |
|-----------|--------|
| **Ubuntu** | 24.04 LTS (Noble Numbat) |
| **Python** | 3.12 |
| **CUDA** | 13.0 |
| **cuDNN** | 9.x |
| **PyTorch** | 2.9+ (cu130) |

### GPU поддержка
PyTorch автоматически устанавливается с поддержкой CUDA 13.0:
```python
import torch
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # Ваша GPU
```

### Предустановленные инструменты
- ✅ JupyterLab
- ✅ Jupyter Server
- ✅ Git, Git LFS

---

## 📁 Структура проекта

```
~/project/
├── README.md                        # Этот файл
├── TypingCat.png                    # Талисман 🐱
└── jupyter/
    ├── 01_setup_and_cpu_test.ipynb  # Установка пакетов + тест CPU
    └── 02_gpu_test.ipynb            # Тест GPU/CUDA
```

### 🚀 С чего начать?
1. Откройте `jupyter/01_setup_and_cpu_test.ipynb`
2. Выполните ячейку установки пакетов
3. Запустите все тесты CPU
4. Откройте `jupyter/02_gpu_test.ipynb` для проверки GPU

---

Удачной работы с данными! 🚀