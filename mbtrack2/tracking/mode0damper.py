import numpy as np
from scipy import signal
from collections import deque
import matplotlib.pyplot as plt

class Mode0DamperFilter:
    """
    数字滤波器类，用于实现Mode-0阻尼器的相位处理
    提供90度相移和增益控制
    """
    def __init__(self, filter_type='differentiator', gain=1.0, history_length=100):
        self.filter_type = filter_type
        self.gain = gain
        self.history_length = history_length
        self.phase_history = deque(maxlen=history_length)
        
        # 初始化滤波器系数
        if filter_type == 'differentiator':
            # 简单差分器实现90度相移
            self.b = np.array([1, -1])  # 分子系数
            self.a = np.array([1, 0])   # 分母系数
        elif filter_type == 'ac_coupled':
            # AC耦合放大器
            alpha = 0.95  # 高通滤波器系数
            self.b = np.array([1, -1])
            self.a = np.array([1, -alpha])
        elif filter_type == 'iir_90deg':
            # IIR滤波器实现更精确的90度相移
            # 设计一个在同步频率处提供90度相移的滤波器
            self.b = np.array([0.5, 0, -0.5])
            self.a = np.array([1, -0.9, 0])
        
        # 滤波器状态
        self.zi = signal.lfiltic(self.b, self.a, [0])
    
    def process(self, phase_error):
        """
        处理相位误差信号
        
        Parameters:
        phase_error (float): 输入的相位误差
        
        Returns:
        float: 处理后的校正信号
        """
        self.phase_history.append(phase_error)
        
        if len(self.phase_history) < 2:
            return 0.0
        
        # 应用数字滤波器
        filtered_signal, self.zi = signal.lfilter(
            self.b, self.a, [phase_error], zi=self.zi
        )
        
        return self.gain * filtered_signal[0]

class Mode0Damper:
    """
    Mode-0阻尼器主类，适配mbtrack2代码库
    集成相位检测、滤波处理和RF腔体调制功能
    """
    def __init__(self, ring, cavity, beam, damper_gain=0.1, filter_type='differentiator'):
        """
        初始化Mode-0阻尼器
        
        Parameters:
        ring: mbtrack2 ring object
        cavity: mbtrack2 RF cavity object  
        beam: mbtrack2 beam object
        damper_gain (float): 阻尼器增益
        filter_type (str): 滤波器类型
        """
        self.ring = ring
        self.cavity = cavity
        self.beam = beam
        self.enabled = True
        
        # 计算同步振荡频率
        self.omega_s = self.compute_synchrotron_frequency()
        
        # 初始化滤波器
        self.filter = Mode0DamperFilter(
            filter_type=filter_type, 
            gain=damper_gain
        )
        
        # 监测数据存储
        self.turn_counter = 0
        self.phase_data = []
        self.correction_data = []
        self.cavity_phase_data = []
        self.beam_phase_data = []
        
    def compute_synchrotron_frequency(self):
        """
        计算同步振荡频率
        
        Returns:
        float: 同步振荡角频率 (rad/s)
        """
        # 从ring和cavity参数计算同步振荡频率
        # omega_s = sqrt(q*V*eta*h / (2*pi*E*beta^2))
        try:
            h = self.ring.h  # 谐波数
            f0 = self.ring.f0  # 回转频率
            eta = getattr(self.ring, 'eta', 0.001)  # 滑移因子，默认值
            E0 = getattr(self.ring, 'E0', 3e9)  # 束流能量，默认3GeV
            
            if hasattr(self.cavity, 'voltage'):
                V_rf = abs(self.cavity.voltage)
            elif hasattr(self.cavity, 'V'):
                V_rf = abs(self.cavity.V)
            else:
                V_rf = 1e6  # 默认1MV
            
            # 计算同步振荡频率
            omega_s_squared = (1.602e-19 * V_rf * eta * h) / (2 * np.pi * E0)
            omega_s = np.sqrt(abs(omega_s_squared)) * 2 * np.pi * f0
            
            return omega_s
        except:
            # 如果无法计算，使用默认值
            return 2 * np.pi * 1000  # 1kHz
    
    def compute_mean_z(self):
        """
        计算所有束团的平均纵向位置
        使用您提供的代码
        """
        # Average over all bunches' longitudinal position (z)
        return np.mean(self.beam.bunch_mean[4, :])
    
    def compute_phase_error(self, mean_tau):
        """
        将平均纵向时间偏移mean_tau (s)转换为RF相位(rad)
        并计算相对于腔体目标相位theta的相位误差
        修正了您的代码
        """
        # RF frequency
        f_rf = self.ring.h * self.ring.f0  # Hz
        
        # convert time offset to phase [rad]
        phi_b = 2 * np.pi * f_rf * mean_tau
        
        # 获取腔体相位
        if hasattr(self.cavity, 'theta'):
            cavity_phase = self.cavity.theta
        elif hasattr(self.cavity, 'phase'):
            cavity_phase = self.cavity.phase
        else:
            cavity_phase = 0.0
        
        # phase error
        delta_phi = phi_b - cavity_phase
        
        # 相位包装到 [-π, π] 范围
        delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
        
        # Debug print (可选)
        # print(f"mean_tau = {mean_tau:.3e} s, beam phase = {phi_b:.3e} rad, "
        #       f"cavity.theta = {cavity_phase:.3e} rad, delta_phi = {delta_phi:.3e} rad")
        
        return delta_phi
    
    def apply_feedback(self):
        """
        应用Mode-0阻尼反馈
        这是主要的接口函数，每个回转周期调用一次
        """
        if not self.enabled:
            return
        
        # 1. 相位检测 - 获取束流平均纵向位置
        mean_z = self.compute_mean_z()  # 获取平均z位置
        
        # 将z位置转换为时间偏移
        # 假设z的单位是米，转换为时间偏移(秒)
        c = 299792458  # 光速 m/s
        mean_tau = mean_z / c
        
        # 2. 计算相位误差
        phase_error = self.compute_phase_error(mean_tau)
        
        # 3. 滤波处理
        correction_signal = self.filter.process(phase_error)
        
        # 4. 调制RF腔体
        self.modulate_cavity(correction_signal)
        
        # 5. 记录数据
        beam_phase = 2 * np.pi * self.ring.h * self.ring.f0 * mean_tau
        cavity_phase = getattr(self.cavity, 'theta', getattr(self.cavity, 'phase', 0.0))
        
        self.phase_data.append(phase_error)
        self.correction_data.append(correction_signal)
        self.beam_phase_data.append(beam_phase)
        self.cavity_phase_data.append(cavity_phase)
        self.turn_counter += 1
    
    def modulate_cavity(self, correction_signal):
        """
        调制RF腔体的相位
        
        Parameters:
        correction_signal (float): 校正信号
        """
        # 方法1: 直接调制腔体相位
        if hasattr(self.cavity, 'theta'):
            self.cavity.theta += correction_signal
        elif hasattr(self.cavity, 'phase'):
            self.cavity.phase += correction_signal
        
        # 方法2: 调制腔体频率（如果支持）
        elif hasattr(self.cavity, 'frequency'):
            df = correction_signal / (2 * np.pi)  # 频率偏移
            self.cavity.frequency += df
        
        # 方法3: 通过电压相位调制
        elif hasattr(self.cavity, 'V') and hasattr(self.cavity, 'phi'):
            # 如果腔体有复数电压表示
            current_magnitude = abs(self.cavity.V)
            current_phase = np.angle(self.cavity.V) if np.iscomplexobj(self.cavity.V) else getattr(self.cavity, 'phi', 0.0)
            new_phase = current_phase + correction_signal
            self.cavity.V = current_magnitude * np.exp(1j * new_phase)
        
        # 方法4: 如果腔体有专门的相位调制接口
        elif hasattr(self.cavity, 'add_phase_modulation'):
            self.cavity.add_phase_modulation(correction_signal)
        
        else:
            print("警告: 无法找到合适的腔体调制接口")
    
    def get_damping_time_constant(self):
        """
        计算理论阻尼时间常数
        
        Returns:
        float: 阻尼时间常数（回转周期数）
        """
        if self.filter.gain > 0 and self.omega_s > 0:
            omega_rev = 2 * np.pi * self.ring.f0
            return 2.0 * omega_rev / (self.filter.gain * self.omega_s)
        else:
            return np.inf
    
    def set_gain(self, new_gain):
        """
        动态调整阻尼器增益
        
        Parameters:
        new_gain (float): 新的增益值
        """
        self.filter.gain = new_gain
    
    def enable(self):
        """启用阻尼器"""
        self.enabled = True
    
    def disable(self):
        """禁用阻尼器"""
        self.enabled = False
    
    def reset(self):
        """重置阻尼器状态"""
        self.turn_counter = 0
        self.phase_data.clear()
        self.correction_data.clear()
        self.beam_phase_data.clear()
        self.cavity_phase_data.clear()
        self.filter.zi = signal.lfiltic(self.filter.b, self.filter.a, [0])
    
    def analyze_performance(self, plot=True):
        """
        分析阻尼器性能
        
        Parameters:
        plot (bool): 是否绘制分析图
        
        Returns:
        dict: 性能指标
        """
        if len(self.phase_data) < 10:
            print("数据不足，无法分析性能")
            return {}
        
        phase_array = np.array(self.phase_data)
        correction_array = np.array(self.correction_data)
        
        # 计算相位振荡的衰减
        phase_envelope = np.abs(signal.hilbert(phase_array - np.mean(phase_array)))
        
        # 拟合指数衰减
        turns = np.arange(len(phase_envelope))
        if np.max(phase_envelope) > 1e-10:
            # 避免log(0)
            safe_envelope = np.maximum(phase_envelope, 1e-10)
            log_envelope = np.log(safe_envelope)
            try:
                decay_fit = np.polyfit(turns, log_envelope, 1)
                measured_damping_rate = -decay_fit[0]
            except:
                measured_damping_rate = 0.0
        else:
            measured_damping_rate = 0.0
        
        performance = {
            'measured_damping_rate': measured_damping_rate,
            'theoretical_damping_time': self.get_damping_time_constant(),
            'final_phase_amplitude': np.std(phase_array[-100:]) if len(phase_array) > 100 else np.std(phase_array),
            'correction_rms': np.std(correction_array),
            'total_turns': len(self.phase_data),
            'synchrotron_frequency': self.omega_s / (2 * np.pi),
            'current_gain': self.filter.gain
        }
        
        if plot:
            self.plot_performance()
        
        return performance
    
    def plot_performance(self):
        """绘制阻尼器性能图表"""
        if len(self.phase_data) < 2:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        turns = np.arange(len(self.phase_data))
        
        # 束流相位误差演化
        ax1.plot(turns, np.array(self.phase_data) * 180/np.pi, 'b-', label='相位误差')
        ax1.set_xlabel('回转数')
        ax1.set_ylabel('相位误差 (度)')
        ax1.set_title('Mode-0阻尼器: 相位误差演化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 校正信号
        ax2.plot(turns, self.correction_data, 'r-', label='校正信号')
        ax2.set_xlabel('回转数')
        ax2.set_ylabel('校正信号 (rad)')
        ax2.set_title('阻尼器校正信号')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 相位振荡包络
        phase_array = np.array(self.phase_data)
        phase_envelope = np.abs(signal.hilbert(phase_array - np.mean(phase_array)))
        ax3.semilogy(turns, phase_envelope * 180/np.pi, 'g-', label='振荡包络')
        ax3.set_xlabel('回转数')
        ax3.set_ylabel('相位振荡包络 (度)')
        ax3.set_title('阻尼效果 (对数标度)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 束流和腔体相位比较
        if len(self.beam_phase_data) > 0 and len(self.cavity_phase_data) > 0:
            ax4.plot(turns, np.array(self.beam_phase_data) * 180/np.pi, 
                    'b-', label='束流相位', alpha=0.7)
            ax4.plot(turns, np.array(self.cavity_phase_data) * 180/np.pi, 
                    'r--', label='腔体相位', alpha=0.7)
            ax4.set_xlabel('回转数')
            ax4.set_ylabel('相位 (度)')
            ax4.set_title('束流与腔体相位比较')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Mode-0阻尼器性能分析 (增益={self.filter.gain:.3f})', 
                     fontsize=14, y=1.02)
        plt.show()

    @classmethod
    def create_for_mbtrack2(cls, ring, cavity, beam, gain=0.05, filter_type='differentiator'):
        """
        为mbtrack2创建Mode-0阻尼器的类方法
        
        Parameters:
        ring: mbtrack2 ring object
        cavity: mbtrack2 cavity object  
        beam: mbtrack2 beam object
        gain (float): 阻尼器增益
        filter_type (str): 滤波器类型
        
        Returns:
        Mode0Damper: 配置好的阻尼器对象
        """
        # 创建Mode-0阻尼器
        damper = cls(ring, cavity, beam, damper_gain=gain, filter_type=filter_type)
        
        print(f"Mode-0阻尼器已创建:")
        print(f"- 同步振荡频率: {damper.omega_s/(2*np.pi):.1f} Hz") 
        print(f"- 理论阻尼时间: {damper.get_damping_time_constant():.1f} 回转周期")
        print(f"- 阻尼器增益: {gain}")
        print(f"- 滤波器类型: {filter_type}")
        
        return damper

# 使用示例:
"""
在mbtrack2主追踪循环中的集成方式:

from tqdm import tqdm

# 初始化阶段 - 使用类方法创建
damper = Mode0Damper.create_for_mbtrack2(ring, RF, bb, gain=0.05)

# 或者直接创建
# damper = Mode0Damper(ring, RF, bb, damper_gain=0.05)

# 主追踪循环
for turn in tqdm(range(nt+1)):
    long.track(bb)  # 正常的mbtrack2纵向追踪
    
    # 在每个回转周期结束时应用Mode-0阻尼
    damper.apply_feedback()
    
    # 可选: 每1000回转监测一次性能
    if turn % 1000 == 0 and turn > 0:
        performance = damper.analyze_performance(plot=False)
        print(f"回转 {turn}: 相位振幅 = {performance['final_phase_amplitude']*180/np.pi:.4f}°")

# 最终分析
final_performance = damper.analyze_performance(plot=True)
"""

# 快速使用的便捷函数
def create_mode0_damper(ring, cavity, beam, gain=0.05):
    """便捷函数，用于快速创建Mode-0阻尼器"""
    return Mode0Damper.create_for_mbtrack2(ring, cavity, beam, gain)