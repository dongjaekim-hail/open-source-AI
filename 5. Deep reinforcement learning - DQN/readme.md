# 0. 소개

강화학습을 학습시키기 위해서는 환경 environment 가 필요합니다.
생각해보세요. 여러분이 여러분의 모델이 있고 (agent) 그것을 학습 / 또는 평가 하려고 한다면 정의된 문제가 필요합니다.


**정의된 문제란?**
```
States, Actions, Reward 이런 것들이 이미 정의되어 나오는 문제들을 의미합니다.
예를 들어 Atari game 같은 경우는 아타리 게임 Rom을 포팅하여 게임을 python 내부로 가져온다면 그게 정의된 문제가 되는 것이죠. 
```

그런 식으로 python 으로 문제를 정의할 때, 환경 environment 를 만들때 우리는 일반적으로 openAI의 gym libray를 이용합니다.

*참고: 과거에는 gym이 openAI 가 관리하던 라이브러리였는데, 이제는 지원이 끊기고 gymnasium 이라는 별도의 라이브러리로 관리되고 있습니다.
다만 아직 과도기적이므로 오류가 많습니다. 따라서 여기서는 그대로 gym을 쓰겠습니다.*

# 1. gym

수업시간에 배웠겠지만 일반적으로 이렇게 사용하면됩니다. https://www.gymlibrary.dev/

버전 따라서 여러가지 이슈가 있으므로 예제에 나온 버전을 그대로 설치해서 활용해보세요.

**여러가지 이슈**
```buildoutcfg
1. 최신 버전에서 Atari rom 의 문제. 저작권은 소멸되었지만 포팅된 롬은 다른 그룹이 했으므로 별도로 다운받아야함
2. 최신 버전에서 gym 자체의 output 갯수가 다른 문제. 버전 업이 되면서 여러가지 기능이 생겨서 그렇습니다. 무시하세요.
```

## 설치
```commandline
pip install gym==0.26.0
pip install gym[all]
pip install gym[accept-rom-license]
```

각각 gym 을 설치, 그리고 gym 에 세부 환경들 설치.
`gym[all]` 의 경우는 모든 환경을 설치하는데, 그 중에는 `gym[atari]` 도 있습니다. 다만 라이센스 agree 를 해야하기 때문에 그 다음 라인을 한번 더 실행해서 완벽히 임포트 해야합니다. 

# 2. DQN 코드

```commandline
pip install tensorflow==2.9.0
```
최신버전해도 됩니다. 다만 혹시 미래에 이 코드 확인했을때 버전 dependency 생길까봐... 

코드는 실제로 보면서 설명할게요. 왜냐하면 이게 graphic 을 그려야하는거라 colab 에서 돌리면 어떻게 돌아가는지 보일 수 가 없습니다. 