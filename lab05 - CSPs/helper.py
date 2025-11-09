import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import time
import math
import statistics

# Thiết lập seed để có thể tái tạo kết quả
np.random.seed(1234)

def random_board(n):
    """Tạo một bàn cờ ngẫu nhiên kích thước n x n. 
    Lưu ý rằng chỉ có một quân hậu duy nhất được đặt trong mỗi cột!"""
    return np.random.randint(0, n, size=n)

def comb2(n): 
    """Tính toán n choose 2 tương đương với math.comb(n, 2)"""
    return n * (n - 1) // 2  # // là phép chia số nguyên

def conflicts(board):
    """Tính toán số lượng xung đột, tức là hàm mục tiêu."""
    n = len(board)
    
    # Đếm số quân hậu trong mỗi hàng
    horizontal_cnt = [0] * n
    # Đếm số quân hậu trong mỗi đường chéo chính (từ trái trên xuống phải dưới)
    diagonal1_cnt = [0] * 2 * n
    # Đếm số quân hậu trong mỗi đường chéo phụ (từ phải trên xuống trái dưới)
    diagonal2_cnt = [0] * 2 * n
    
    for i in range(n):
        horizontal_cnt[board[i]] += 1
        diagonal1_cnt[i + board[i]] += 1
        diagonal2_cnt[i - board[i] + n] += 1
    
    # Tính tổng số xung đột từ tất cả các hàng và đường chéo
    return sum(map(comb2, horizontal_cnt + diagonal1_cnt + diagonal2_cnt))

def show_board(board, cols=['white', 'gray'], fontsize=48):
    """Hiển thị bàn cờ"""
    n = len(board)
    
    # Tạo hiển thị bàn cờ
    display = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if ((i + j) % 2) != 0:
                display[i, j] = 1
    
    cmap = colors.ListedColormap(cols)
    fig, ax = plt.subplots()
    ax.imshow(display, cmap=cmap, 
              norm=colors.BoundaryNorm(range(len(cols) + 1), cmap.N))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Đặt các quân hậu. Lưu ý: Unicode u265B là quân hậu đen
    for j in range(n):
        plt.text(j, board[j], u"\u265B", fontsize=fontsize, 
                 horizontalalignment='center',
                 verticalalignment='center')
    
    print(f"Bàn cờ với {conflicts(board)} xung đột.")
    plt.show()

def get_neighbors(board):
    """Tạo tất cả các hàng xóm có thể của bàn cờ hiện tại.
    Mỗi hàng xóm là kết quả của việc di chuyển một quân hậu đến một hàng khác trong cột của nó."""
    neighbors = []
    n = len(board)
    
    for col in range(n):
        for row in range(n):
            if row != board[col]:  # Chỉ tạo hàng xóm nếu hàng khác với hàng hiện tại
                neighbor = board.copy()
                neighbor[col] = row
                neighbors.append(neighbor)
    
    return neighbors

def get_random_neighbor(board):
    """Tạo một hàng xóm ngẫu nhiên của bàn cờ hiện tại."""
    n = len(board)
    neighbor = board.copy()
    
    # Chọn ngẫu nhiên một cột và một hàng mới
    col = random.randint(0, n - 1)
    new_row = random.randint(0, n - 1)
    
    # Đảm bảo hàng mới khác với hàng hiện tại
    while new_row == board[col]:
        new_row = random.randint(0, n - 1)
    
    neighbor[col] = new_row
    return neighbor

def is_goal(board):
    """Kiểm tra xem bàn cờ có phải là giải pháp hay không (không có xung đột)."""
    return conflicts(board) == 0

def measure_time(func, *args, **kwargs):
    """Đo thời gian thực thi của một hàm."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def steepest_ascent_hill_climbing(board, max_iterations=1000):
    """
    Thuật toán leo đồi steepest ascent cho bài toán n-Hậu.
    
    Args:
        board: Bàn cờ ban đầu
        max_iterations: Số lần lặp tối đa
    
    Returns:
        tuple: (bàn cờ cuối cùng, số lần lặp, lịch sử xung đột)
    """
    current_board = board.copy()
    conflicts_history = [conflicts(current_board)]
    iterations = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Kiểm tra xem đã tìm thấy giải pháp chưa
        if is_goal(current_board):
            break
            
        # Lấy tất cả các hàng xóm
        neighbors = get_neighbors(current_board)
        
        # Tìm hàng xóm tốt nhất (có ít xung đột nhất)
        best_neighbor = None
        best_conflicts = conflicts(current_board)
        
        for neighbor in neighbors:
            neighbor_conflicts = conflicts(neighbor)
            if neighbor_conflicts < best_conflicts:
                best_conflicts = neighbor_conflicts
                best_neighbor = neighbor
        
        # Nếu không có hàng xóm nào tốt hơn, đã đạt cực trị cục bộ
        if best_neighbor is None:
            break
            
        # Di chuyển đến hàng xóm tốt nhất
        current_board = best_neighbor
        conflicts_history.append(best_conflicts)
    
    return current_board, iterations, conflicts_history

def stochastic_hill_climbing_1(board, max_iterations=1000):
    """
    Thuật toán leo đồi ngẫu nhiên 1: Chọn ngẫu nhiên từ tất cả các bước di chuyển lên dốc.
    
    Args:
        board: Bàn cờ ban đầu
        max_iterations: Số lần lặp tối đa
    
    Returns:
        tuple: (bàn cờ cuối cùng, số lần lặp, lịch sử xung đột)
    """
    current_board = board.copy()
    conflicts_history = [conflicts(current_board)]
    iterations = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Kiểm tra xem đã tìm thấy giải pháp chưa
        if is_goal(current_board):
            break
            
        # Lấy tất cả các hàng xóm
        neighbors = get_neighbors(current_board)
        current_conflicts = conflicts(current_board)
        
        # Lọc ra các hàng xóm tốt hơn (uphill moves)
        uphill_moves = [neighbor for neighbor in neighbors 
                       if conflicts(neighbor) < current_conflicts]
        
        # Nếu không có bước di chuyển lên dốc nào, đã đạt cực trị cục bộ
        if not uphill_moves:
            break
            
        # Chọn ngẫu nhiên một bước di chuyển lên dốc
        current_board = random.choice(uphill_moves)
        conflicts_history.append(conflicts(current_board))
    
    return current_board, iterations, conflicts_history

def stochastic_hill_climbing_2(board, max_iterations=1000, max_no_improvement=50):
    """
    Thuật toán leo đồi ngẫu nhiên 2 (First-choice hill climbing): 
    Tạo một hàng xóm ngẫu nhiên và chấp nhận nếu tốt hơn.
    
    Args:
        board: Bàn cờ ban đầu
        max_iterations: Số lần lặp tối đa
        max_no_improvement: Số lần thử không cải thiện tối đa trước khi dừng
    
    Returns:
        tuple: (bàn cờ cuối cùng, số lần lặp, lịch sử xung đột)
    """
    current_board = board.copy()
    conflicts_history = [conflicts(current_board)]
    iterations = 0
    no_improvement_count = 0
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Kiểm tra xem đã tìm thấy giải pháp chưa
        if is_goal(current_board):
            break
            
        # Tạo một hàng xóm ngẫu nhiên
        neighbor = get_random_neighbor(current_board)
        current_conflicts = conflicts(current_board)
        neighbor_conflicts = conflicts(neighbor)
        
        # Nếu hàng xóm tốt hơn, chấp nhận nó
        if neighbor_conflicts < current_conflicts:
            current_board = neighbor
            conflicts_history.append(neighbor_conflicts)
            no_improvement_count = 0
        else:
            # Không cải thiện
            no_improvement_count += 1
            
            # Nếu không cải thiện quá nhiều lần, có thể đã đạt cực trị cục bộ
            if no_improvement_count >= max_no_improvement:
                break
    
    return current_board, iterations, conflicts_history

def hill_climbing_with_random_restarts(algorithm_func, board_size, max_restarts=100, max_iterations_per_restart=1000):
    """
    Chạy thuật toán leo đồi với khởi động lại ngẫu nhiên.
    
    Args:
        algorithm_func: Hàm thuật toán leo đồi
        board_size: Kích thước bàn cờ
        max_restarts: Số lần khởi động lại tối đa
        max_iterations_per_restart: Số lần lặp tối đa cho mỗi lần khởi động
    
    Returns:
        tuple: (bàn cờ tốt nhất, số lần khởi động, tổng số lần lặp, lịch sử)
    """
    best_board = None
    best_conflicts = float('inf')
    total_iterations = 0
    restarts_needed = 0
    all_histories = []
    
    for restart in range(max_restarts):
        restarts_needed = restart + 1
        
        # Tạo bàn cờ ngẫu nhiên mới
        initial_board = random_board(board_size)
        
        # Chạy thuật toán
        final_board, iterations, history = algorithm_func(initial_board, max_iterations_per_restart)
        total_iterations += iterations
        all_histories.append(history)
        
        # Kiểm tra xem có tốt hơn không
        final_conflicts = conflicts(final_board)
        if final_conflicts < best_conflicts:
            best_conflicts = final_conflicts
            best_board = final_board
            
            # Nếu tìm thấy giải pháp tối ưu, dừng lại
            if final_conflicts == 0:
                break
    
    return best_board, restarts_needed, total_iterations, all_histories

def simulated_annealing(board, initial_temp=1000, cooling_rate=0.95, min_temp=0.1, max_iterations=10000):
    """
    Thuật toán luyện kim mô phỏng cho bài toán n-Hậu.
    
    Args:
        board: Bàn cờ ban đầu
        initial_temp: Nhiệt độ ban đầu
        cooling_rate: Tỷ lệ làm mát
        min_temp: Nhiệt độ tối thiểu
        max_iterations: Số lần lặp tối đa
    
    Returns:
        tuple: (bàn cờ cuối cùng, số lần lặp, lịch sử xung đột, lịch sử nhiệt độ)
    """
    current_board = board.copy()
    current_conflicts = conflicts(current_board)
    
    conflicts_history = [current_conflicts]
    temperature_history = [initial_temp]
    iterations = 0
    
    temp = initial_temp
    
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Kiểm tra xem đã tìm thấy giải pháp chưa
        if current_conflicts == 0:
            break
            
        # Tạo một hàng xóm ngẫu nhiên
        neighbor = get_random_neighbor(current_board)
        neighbor_conflicts = conflicts(neighbor)
        
        # Tính toán sự khác biệt về xung đột
        delta = neighbor_conflicts - current_conflicts
        
        # Nếu hàng xóm tốt hơn hoặc được chấp nhận theo xác suất
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_board = neighbor
            current_conflicts = neighbor_conflicts
        
        # Cập nhật lịch sử
        conflicts_history.append(current_conflicts)
        temperature_history.append(temp)
        
        # Giảm nhiệt độ
        temp *= cooling_rate
        
        # Dừng nếu nhiệt độ quá thấp
        if temp < min_temp:
            break
    
    return current_board, iterations, conflicts_history, temperature_history

def run_algorithm_multiple_times(algorithm_func, board_size, num_runs=100, max_iterations=1000):
    """
    Chạy một thuật toán nhiều lần và thu thập thống kê.
    
    Args:
        algorithm_func: Hàm thuật toán
        board_size: Kích thước bàn cờ
        num_runs: Số lần chạy
        max_iterations: Số lần lặp tối đa cho mỗi lần chạy
    
    Returns:
        dict: Thống kê về thời gian, số xung đột, tỷ lệ thành công
    """
    runtimes = []
    final_conflicts = []
    iterations_count = []
    success_count = 0
    
    for run in range(num_runs):
        # Tạo bàn cờ ngẫu nhiên
        board = random_board(board_size)
        
        # Đo thời gian
        start_time = time.time()
        
        if algorithm_func == simulated_annealing:
            final_board, iterations, conflicts_hist, temp_hist = algorithm_func(
                board, max_iterations=max_iterations
            )
        else:
            final_board, iterations, conflicts_hist = algorithm_func(
                board, max_iterations=max_iterations
            )
        
        end_time = time.time()
        
        # Thu thập dữ liệu
        runtime = end_time - start_time
        runtimes.append(runtime)
        final_conflicts.append(conflicts(final_board))
        iterations_count.append(iterations)
        
        if conflicts(final_board) == 0:
            success_count += 1
    
    return {
        'avg_runtime': statistics.mean(runtimes),
        'avg_conflicts': statistics.mean(final_conflicts),
        'avg_iterations': statistics.mean(iterations_count),
        'success_rate': success_count / num_runs,
        'runtimes': runtimes,
        'final_conflicts': final_conflicts,
        'iterations': iterations_count
    }
