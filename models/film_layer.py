# 컨트롤러 (FiLM Generator)
import torch
import torch.nn as nn

# ==============================================================================
# 🎛️ The Controller: Multi-Stage FiLM Generator
# 역할: LLM의 특징(Feature)을 받아 VETNet의 각 층을 제어할 파라미터(Scale, Shift) 생성
# 핵심: Zero-Initialization (학습 초기 안정성 확보)
# ==============================================================================

class FiLMGenerator(nn.Module):
    def __init__(self, input_dim=4096, vetnet_channels=[64, 128, 256, 512]):
        """
        Args:
            input_dim: Pixel Decoder의 출력 차원 (기본 4096)
            vetnet_channels: VETNet의 각 스테이지별 채널 수 리스트
        """
        super().__init__()
        self.input_dim = input_dim
        self.channels = vetnet_channels
        
        # 각 채널별로 별도의 제어 헤드(Head)를 생성합니다.
        # 예: 64채널 레이어를 제어하려면 -> 64(Scale) + 64(Shift) = 128개 출력 필요
        self.heads = nn.ModuleList()
        
        for ch in vetnet_channels:
            # Simple Projection: Input -> Hidden -> Output(2*ch)
            # 복잡한 구조보다는 선형 변환이 신호 전달에 유리함
            head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 4),
                nn.GELU(),
                nn.Linear(input_dim // 4, ch * 2) # Output: [gamma, beta]
            )
            
            # 🔧 Zero-Initialization 적용
            # 마지막 레이어(Linear)를 찾아 가중치와 편향을 0으로 초기화
            self._zero_init_head(head)
            self.heads.append(head)

    def _zero_init_head(self, head_module):
        """
        명세서 핵심: 마지막 레이어를 0으로 초기화하여, 
        학습 초기에는 gamma=0, beta=0이 되도록 함.
        결과적으로 초기 출력은 Identity(원래 특징 그대로)가 됨.
        """
        last_linear = head_module[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, llm_features):
        """
        Args:
            llm_features: [Batch, 257, 4096] (From Pixel Decoder)
            
        Returns:
            film_signals: List of tuples [(gamma, beta), ...] for each layer
        """
        # 1. Global Context Extraction
        # 전체 이미지의 문맥을 담고 있는 CLS 토큰(인덱스 0)을 사용
        # [B, 257, 4096] -> [B, 4096]
        global_context = llm_features[:, 0, :] 
        
        film_signals = []
        
        # 2. Generate Parameters for each layer
        for head in self.heads:
            # [B, 4096] -> [B, ch * 2]
            params = head(global_context)
            
            # Split into Gamma (Scale) and Beta (Shift)
            # chunk(2, dim=1) -> [B, ch], [B, ch]
            gamma, beta = params.chunk(2, dim=1)
            
            # VETNet(Conv2d)에 적용하기 위해 차원 확장: [B, ch, 1, 1]
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
            film_signals.append((gamma, beta))
            
        return film_signals


# ==============================================================================
# 🔌 The Receiver: FiLM Layer (Applied inside VETNet)
# 역할: 실제 VETNet 내부에서 Feature Map을 변조하는 연산 수행
# 수식: Feature_new = Feature_old * (1 + gamma) + beta
# ==============================================================================

class FiLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # 파라미터가 없으므로 __init__에서 할 일은 없음
        pass

    def forward(self, x, film_signal):
        """
        Args:
            x: [Batch, Channel, H, W] (VETNet Feature Map)
            film_signal: (gamma, beta) tuple from FiLMGenerator
        """
        gamma, beta = film_signal
        
        # 1. Scale (Multiplication)
        # (1 + gamma)를 곱해주는 이유는 gamma가 0일 때 
        # "1배(변화 없음)"가 되게 하기 위함 (Residual Learning 관점)
        x = x * (1 + gamma)
        
        # 2. Shift (Addition)
        x = x + beta
        
        return x