#!/usr/bin/env python3
"""
Spleeter 安装和管理脚本

Spleeter 是 Deezer 开源的人声分离工具，基于深度学习，
分离效果比传统算法好很多。
"""

import subprocess
import sys
import os


def check_spleeter():
    """检查 Spleeter 是否已安装"""
    try:
        import spleeter
        return True
    except ImportError:
        return False


def install_spleeter():
    """安装 Spleeter"""
    print("🎵 正在安装 Spleeter...")
    print("   这是一个深度学习模型，需要下载约 1.5GB 的数据文件")
    print("   安装可能需要几分钟时间...")
    print()
    
    try:
        # 安装 spleeter
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "spleeter"
        ])
        
        print()
        print("✅ Spleeter 安装成功!")
        print()
        print("📝 使用说明:")
        print("   from note_get import SpleeterWrapper")
        print("   wrapper = SpleeterWrapper()")
        print("   vocal = wrapper.extract_vocal(audio, sample_rate)")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        print()
        print("💡 可能的解决方案:")
        print("   1. 确保网络连接正常")
        print("   2. 使用管理员权限运行")
        print("   3. 手动安装: pip install spleeter")
        print()
        return False


def install_spleeter_gpu():
    """安装 GPU 版本的 Spleeter (需要 NVIDIA GPU)"""
    print("🎵 正在安装 Spleeter (GPU版本)...")
    print("   注意: 需要 NVIDIA GPU 和 CUDA 支持")
    print()
    
    try:
        # 先安装 tensorflow-gpu
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "tensorflow-gpu"
        ])
        
        # 安装 spleeter
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "spleeter"
        ])
        
        print()
        print("✅ Spleeter (GPU版) 安装成功!")
        print("   分离速度将比 CPU 版本快 5-10 倍")
        print()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False


def uninstall_spleeter():
    """卸载 Spleeter"""
    print("🗑️  正在卸载 Spleeter...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y", "spleeter"
        ])
        
        print("✅ Spleeter 已卸载")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 卸载失败: {e}")
        return False


def test_spleeter():
    """测试 Spleeter 是否工作正常"""
    print("🧪 测试 Spleeter...")
    
    try:
        from spleeter.separator import Separator
        
        # 尝试创建分离器 (这会触发模型下载)
        print("   加载模型...")
        separator = Separator('spleeter:2stems')
        
        print("✅ Spleeter 工作正常!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Spleeter 安装管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python install_spleeter.py install       # 安装 Spleeter
  python install_spleeter.py install-gpu   # 安装 GPU 版本
  python install_spleeter.py check         # 检查是否已安装
  python install_spleeter.py test          # 测试 Spleeter
  python install_spleeter.py uninstall     # 卸载 Spleeter
        """
    )
    
    parser.add_argument(
        "command",
        choices=["install", "install-gpu", "check", "test", "uninstall"],
        help="要执行的命令"
    )
    
    args = parser.parse_args()
    
    if args.command == "install":
        if check_spleeter():
            print("✅ Spleeter 已经安装")
            response = input("是否重新安装? (y/N): ")
            if response.lower() != 'y':
                return
        install_spleeter()
        
    elif args.command == "install-gpu":
        if check_spleeter():
            print("✅ Spleeter 已经安装")
            response = input("是否重新安装 GPU 版本? (y/N): ")
            if response.lower() != 'y':
                return
        install_spleeter_gpu()
        
    elif args.command == "check":
        if check_spleeter():
            print("✅ Spleeter 已安装")
            try:
                import spleeter
                print(f"   版本: {spleeter.__version__}")
            except:
                pass
        else:
            print("❌ Spleeter 未安装")
            print("   运行: python install_spleeter.py install")
            
    elif args.command == "test":
        if not check_spleeter():
            print("❌ Spleeter 未安装")
            print("   请先运行: python install_spleeter.py install")
            return
        test_spleeter()
        
    elif args.command == "uninstall":
        if not check_spleeter():
            print("❌ Spleeter 未安装")
            return
        response = input("确定要卸载 Spleeter? (y/N): ")
        if response.lower() == 'y':
            uninstall_spleeter()


if __name__ == "__main__":
    main()
