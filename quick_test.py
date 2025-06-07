#!/usr/bin/env python3
"""
Quick TorchSlim functionality test
"""

def main():
    print("🔥 TorchSlim Quick Test")
    print("=" * 30)
    
    try:
        # Import
        from torchslim import TorchSlim, CompressionConfig, create_test_model
        import torch
        print("✅ Imports successful")
        
        # Create model
        model = create_test_model("mlp", input_size=20, hidden_sizes=[10], output_size=5)
        print(f"✅ Model created ({sum(p.numel() for p in model.parameters())} params)")
        
        # Configure compression
        config = CompressionConfig()
        config.add_method("svd", rank_ratio=0.5)
        print("✅ Configuration created")
        
        # Compress
        compressor = TorchSlim(config)
        compressed_model = compressor.compress_model(model)
        ratio = compressor.get_compression_report()['summary']['compression_ratio']
        print(f"✅ Compression successful ({ratio:.2f}x)")
        
        # Test
        x = torch.randn(1, 20)
        y1 = model(x)
        y2 = compressed_model(x)
        mse = torch.mean((y1 - y2)**2).item()
        print(f"✅ Models work (MSE: {mse:.6f})")
        
        print("\n🎉 All tests passed! TorchSlim is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
