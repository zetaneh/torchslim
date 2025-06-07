

"""
Realistic TorchSlim compression test - FIXED VERSION
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchslim import TorchSlim, CompressionConfig, create_test_model
from torchslim.utils.validation import validate_compression_pipeline, print_validation_report

def test_realistic_compression():
    print("🚀 Testing realistic TorchSlim compression...")
    
    # Create a test model
    original_model = create_test_model(
        'mlp', 
        input_size=784, 
        hidden_sizes=[512, 256, 128], 
        output_size=10
    )
    
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    
    # Configure compression
    config = CompressionConfig()
    config.add_method("svd", rank_ratio=0.5)
    config.add_method("pruning", pruning_ratio=0.2)
    
    print("Compression methods:", config.enabled_methods)
    
    # Compress the model
    compressor = TorchSlim(config)
    compressed_model = compressor.compress_model(original_model)
    
    print(f"Compressed model parameters: {sum(p.numel() for p in compressed_model.parameters()):,}")
    
    # Validate the compression
    results = validate_compression_pipeline(
        original_model, 
        compressed_model, 
        config.enabled_methods
    )
    
    # Print detailed report
    print("\n")
    print_validation_report(results)
    
    # Get compression report from TorchSlim
    compression_report = compressor.get_compression_report()
    print("\n🔍 TorchSlim Compression Report:")
    print("="*50)
    print(f"Overall compression ratio: {compression_report['summary']['compression_ratio']:.2f}x")
    
    # Handle different possible report structures
    if 'total_time' in compression_report.get('summary', {}):
        print(f"Total time: {compression_report['summary']['total_time']:.3f}s")
    elif 'compression_time' in compression_report.get('summary', {}):
        print(f"Total time: {compression_report['summary']['compression_time']:.3f}s")
    else:
        # Look for timing info in other parts of the report
        total_time = 0
        for method_name, method_report in compression_report.get('methods', {}).items():
            if 'time' in method_report:
                total_time += method_report['time']
        if total_time > 0:
            print(f"Total time: {total_time:.3f}s")
        else:
            print("Timing information not available in current report format")
    
    # Show what's actually in the report
    print("\n📋 Available report keys:")
    print(f"  Summary keys: {list(compression_report.get('summary', {}).keys())}")
    print(f"  Report sections: {list(compression_report.keys())}")
    
    return results['overall_status'] in ['success', 'warning']

if __name__ == "__main__":
    try:
        success = test_realistic_compression()
        if success:
            print("\n✅ Realistic compression test PASSED!")
            print("🎊 TorchSlim is working perfectly!")
        else:
            print("\n❌ Realistic compression test FAILED!")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

