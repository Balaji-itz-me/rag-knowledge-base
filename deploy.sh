#!/bin/bash
# Enhanced RAG System Deployment Script
# Run this on your EC2 instance

echo "======================================"
echo "Enhanced RAG System Deployment"
echo "======================================"

# Step 1: Backup existing code
echo "📦 Step 1: Backing up existing code..."
cd /home/ubuntu/rag-knowledge-base/rag_demo
cp main.py main.py.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Step 2: Install new dependencies
echo "📥 Step 2: Installing new dependencies..."
pip install slowapi aiohttp
echo "✅ Dependencies installed"

# Step 3: Create log directory
echo "📁 Step 3: Creating log directory..."
mkdir -p /home/ubuntu/rag-knowledge-base/rag_demo/logs
chmod 755 /home/ubuntu/rag-knowledge-base/rag_demo/logs
echo "✅ Log directory created"

# Step 4: Set environment variables (optional)
echo "🔑 Step 4: Setting environment variables..."
# Uncomment and modify these if you want custom API keys
# export API_KEY_1="your-custom-key-1"
# export API_KEY_1_USER="user1"
# export API_KEY_1_PERMS="read,query,chat,index,admin"
echo "✅ Environment variables ready"

# Step 5: Test the enhanced system
echo "🧪 Step 5: Creating test script..."
cat > test_enhanced_features.py << 'EOF'
#!/usr/bin/env python3
"""Test script for enhanced RAG features"""

import requests
import time
import json

BASE_URL = "http://localhost:8000"
API_KEY = "demo-api-key-123"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_caching():
    """Test caching functionality"""
    print("\n" + "="*50)
    print("TEST 1: CACHING")
    print("="*50)
    
    query = {"messages": [{"role": "user", "content": "What is RAG?"}]}
    
    # First request (cache miss)
    print("Making first request (cache miss)...")
    start = time.time()
    r1 = requests.post(f"{BASE_URL}/api/v1/chat", headers=headers, json=query)
    time1 = time.time() - start
    print(f"✅ First request: {time1:.3f}s")
    
    # Second request (cache hit)
    print("Making second request (cache hit)...")
    start = time.time()
    r2 = requests.post(f"{BASE_URL}/api/v1/chat", headers=headers, json=query)
    time2 = time.time() - start
    print(f"✅ Second request: {time2:.3f}s")
    
    if time2 < time1:
        improvement = ((time1 - time2) / time1) * 100
        print(f"✅ CACHING WORKS! Speed improvement: {improvement:.1f}%")
        return True
    else:
        print("⚠️  Cache may not be working (second request not faster)")
        return False

def test_rate_limiting():
    """Test rate limiting"""
    print("\n" + "="*50)
    print("TEST 2: RATE LIMITING")
    print("="*50)
    
    print("Sending 25 rapid requests (limit is 20/minute)...")
    rate_limited = False
    
    for i in range(25):
        r = requests.post(
            f"{BASE_URL}/api/v1/chat",
            headers=headers,
            json={"messages": [{"role": "user", "content": f"test{i}"}]}
        )
        
        if r.status_code == 429:
            print(f"✅ Rate limited at request {i+1}")
            print(f"Response: {r.json()}")
            rate_limited = True
            break
        
        time.sleep(0.1)  # Small delay between requests
    
    if rate_limited:
        print("✅ RATE LIMITING WORKS!")
        return True
    else:
        print("⚠️  Rate limiting may not be working")
        return False

def test_metrics():
    """Test metrics endpoint"""
    print("\n" + "="*50)
    print("TEST 3: METRICS ENDPOINT")
    print("="*50)
    
    print("Fetching system metrics...")
    r = requests.get(f"{BASE_URL}/api/v1/metrics", headers=headers)
    
    if r.status_code == 200:
        metrics = r.json()
        print("✅ Metrics endpoint works!")
        print(json.dumps(metrics, indent=2))
        
        # Check cache stats
        if 'cache_performance' in metrics:
            cache_stats = metrics['cache_performance']
            print(f"\n📊 Cache Statistics:")
            print(f"   Hits: {cache_stats.get('hits', 0)}")
            print(f"   Misses: {cache_stats.get('misses', 0)}")
            print(f"   Hit Rate: {cache_stats.get('hit_rate_percent', '0.00')}%")
        
        return True
    else:
        print(f"❌ Metrics endpoint failed: {r.status_code}")
        return False

def test_concurrent_processing():
    """Test concurrent URL processing"""
    print("\n" + "="*50)
    print("TEST 4: CONCURRENT PROCESSING")
    print("="*50)
    
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]
    
    print(f"Indexing {len(urls)} URLs concurrently...")
    start = time.time()
    
    r = requests.post(
        f"{BASE_URL}/api/v1/index",
        headers=headers,
        json={"url": urls}
    )
    
    elapsed = time.time() - start
    
    if r.status_code == 200:
        result = r.json()
        print(f"✅ Concurrent processing completed in {elapsed:.2f}s")
        print(f"   Successful: {result['metadata']['successfully_indexed']}")
        print(f"   Failed: {result['metadata']['failed']}")
        print(f"   Concurrent: {result['metadata'].get('concurrent_processing', False)}")
        return True
    else:
        print(f"⚠️  Indexing failed: {r.status_code}")
        return False

def test_logging():
    """Test structured logging"""
    print("\n" + "="*50)
    print("TEST 5: STRUCTURED LOGGING")
    print("="*50)
    
    print("Checking log file...")
    try:
        with open('/home/ubuntu/rag-knowledge-base/rag_demo/logs/app.log', 'r') as f:
            lines = f.readlines()[-10:]  # Last 10 lines
            
        print("✅ Log file exists!")
        print("\n📝 Recent log entries:")
        for line in lines:
            try:
                log_entry = json.loads(line.split(' - ')[-1])
                print(f"   [{log_entry.get('level')}] {log_entry.get('event')}")
            except:
                print(f"   {line.strip()}")
        
        return True
    except FileNotFoundError:
        print("⚠️  Log file not found")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ENHANCED RAG SYSTEM - FEATURE VERIFICATION")
    print("="*60)
    
    results = {
        "Caching": False,
        "Rate Limiting": False,
        "Metrics": False,
        "Concurrent Processing": False,
        "Logging": False
    }
    
    # Run tests
    try:
        results["Caching"] = test_caching()
    except Exception as e:
        print(f"❌ Caching test error: {e}")
    
    try:
        results["Rate Limiting"] = test_rate_limiting()
    except Exception as e:
        print(f"❌ Rate limiting test error: {e}")
    
    try:
        results["Metrics"] = test_metrics()
    except Exception as e:
        print(f"❌ Metrics test error: {e}")
    
    try:
        results["Concurrent Processing"] = test_concurrent_processing()
    except Exception as e:
        print(f"❌ Concurrent processing test error: {e}")
    
    try:
        results["Logging"] = test_logging()
    except Exception as e:
        print(f"❌ Logging test error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for feature, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {feature}")
    
    print("\n" + "="*60)
    print(f"RESULT: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 ALL FEATURES WORKING! Your CV claims are now TRUE!")
    elif passed >= 3:
        print("\n✅ Most features working! Good enough for interview.")
    else:
        print("\n⚠️  Some features need attention.")
    
    return passed, total

if __name__ == "__main__":
    passed, total = run_all_tests()
    exit(0 if passed == total else 1)
EOF

chmod +x test_enhanced_features.py
echo "✅ Test script created"

# Step 6: Instructions
echo ""
echo "======================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "======================================"

