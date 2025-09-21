class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        total_moves = m + n - 2
        k = min(m - 1, n - 1)
        result = 1
        for i in range(1, k + 1):
            result = result * (total_moves - k + i) // i
        return result
          
