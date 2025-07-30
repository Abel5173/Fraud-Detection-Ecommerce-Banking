#!/usr/bin/env python3
"""
Test script to verify installation and dependencies for Fraud Detection System
"""

import sys
import os
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("=== TESTING PACKAGE IMPORTS ===")
    
    required_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'sklearn',
        'imblearn',
        'lightgbm',
        'xgboost',
        'shap',
        'joblib'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_data_access():
    """Test if data files are accessible"""
    print("\n=== TESTING DATA ACCESS ===")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    required_files = [
        "Fraud_Data.csv",
        "creditcard.csv", 
        "IpAddress_to_Country.csv"
    ]
    
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file} ({file_size:.1f} MB)")
        else:
            print(f"❌ {file}: Not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        print("Please ensure all data files are in the data/ directory")
        return False
    else:
        print("\n✅ All data files found!")
        return True

def test_directory_structure():
    """Test if required directories exist and can be created"""
    print("\n=== TESTING DIRECTORY STRUCTURE ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_dirs = [
        "data",
        "outputs", 
        "scripts",
        "notebooks"
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/: Not found")
            missing_dirs.append(dir_name)
    
    # Test if outputs directory can be created
    outputs_dir = os.path.join(base_dir, "outputs")
    try:
        os.makedirs(outputs_dir, exist_ok=True)
        print("✅ outputs/ directory created/accessible")
    except Exception as e:
        print(f"❌ Cannot create outputs/ directory: {e}")
        return False
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("\n✅ All required directories found!")
        return True

def test_basic_functionality():
    """Test basic functionality with sample data"""
    print("\n=== TESTING BASIC FUNCTIONALITY ===")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'class': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        })
        
        # Test preprocessing
        scaler = StandardScaler()
        X = sample_data.drop('class', axis=1)
        y = sample_data['class']
        
        X_scaled = scaler.fit_transform(X)
        
        print("✅ Sample data creation and preprocessing")
        print(f"   - Data shape: {sample_data.shape}")
        print(f"   - Class distribution: {y.value_counts().to_dict()}")
        print(f"   - Scaled features shape: {X_scaled.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 FRAUD DETECTION SYSTEM - INSTALLATION TEST")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Access", test_data_access), 
        ("Directory Structure", test_directory_structure),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run Task 1: python scripts/fraud_detection_task1.py")
        print("2. Run Task 2: python scripts/fraud_detection_task2.py") 
        print("3. Run Task 3: python scripts/fraud_detection_task3.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 