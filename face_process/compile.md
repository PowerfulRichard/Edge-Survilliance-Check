
# 编译方法
mkdir -p build && cd build
cmake ..
make            # 全部编译
# 或
make face_timer  # 只编译计时程序
make face_mini   # 只编译精简程序
# 路径
src/only  	# face_mini源文件
src/time   	# face_timer源文件
