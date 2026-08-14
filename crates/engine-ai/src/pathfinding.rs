//! A* pathfinding on grids and graphs.

use glam::IVec2;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A 2D grid: `true` = blocked, `false` = walkable. Origin (0,0) is top-left.
pub struct Grid {
    pub width: i32,
    pub height: i32,
    pub cells: Vec<bool>,
}

impl Grid {
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            height,
            cells: vec![false; (width * height) as usize],
        }
    }

    pub fn set(&mut self, x: i32, y: i32, blocked: bool) {
        if let Some(i) = self.index(x, y) {
            self.cells[i] = blocked;
        }
    }

    pub fn is_blocked(&self, x: i32, y: i32) -> bool {
        self.index(x, y)
            .map(|i| self.cells[i])
            .unwrap_or(true)
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        if x < 0 || y < 0 || x >= self.width || y >= self.height {
            None
        } else {
            Some((y * self.width + x) as usize)
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct Node {
    cost: u32,
    pos: IVec2,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so reverse the ordering for min-heap behavior.
        // IVec2 isn't Ord, so we compare by (x, y) tuple as a tiebreaker.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| (other.pos.x, other.pos.y).cmp(&(self.pos.x, self.pos.y)))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 4-directional neighbors (Manhattan movement).
const DIRS_4: [IVec2; 4] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
];

/// 8-directional neighbors (Chebyshev movement — diagonal allowed).
const DIRS_8: [IVec2; 8] = [
    IVec2::new(1, 0),
    IVec2::new(-1, 0),
    IVec2::new(0, 1),
    IVec2::new(0, -1),
    IVec2::new(1, 1),
    IVec2::new(1, -1),
    IVec2::new(-1, 1),
    IVec2::new(-1, -1),
];

/// A* on a 2D grid. Returns the path from `start` to `goal` (inclusive of both),
/// or `None` if no path exists. Uses 8-directional movement by default.
pub fn astar_grid(grid: &Grid, start: IVec2, goal: IVec2) -> Option<Vec<IVec2>> {
    if grid.is_blocked(start.x, start.y) || grid.is_blocked(goal.x, goal.y) {
        return None;
    }
    let mut open: BinaryHeap<Node> = BinaryHeap::new();
    let mut came_from: HashMap<IVec2, IVec2> = HashMap::new();
    let mut g_score: HashMap<IVec2, u32> = HashMap::new();
    g_score.insert(start, 0);
    open.push(Node {
        cost: heuristic(start, goal),
        pos: start,
    });

    while let Some(Node { pos, .. }) = open.pop() {
        if pos == goal {
            // Reconstruct path.
            let mut path = vec![pos];
            let mut cur = pos;
            while let Some(&prev) = came_from.get(&cur) {
                path.push(prev);
                cur = prev;
            }
            path.reverse();
            return Some(path);
        }
        let cur_g = *g_score.get(&pos).unwrap_or(&u32::MAX);
        for d in &DIRS_8 {
            let next = pos + *d;
            if grid.is_blocked(next.x, next.y) {
                continue;
            }
            // Prevent diagonal cutting through walls.
            if d.x != 0 && d.y != 0 {
                if grid.is_blocked(pos.x + d.x, pos.y) || grid.is_blocked(pos.x, pos.y + d.y) {
                    continue;
                }
            }
            let step_cost: u32 = if d.x != 0 && d.y != 0 { 14 } else { 10 }; // sqrt(2) ~ 1.4
            let tentative_g = cur_g.saturating_add(step_cost);
            let existing = *g_score.get(&next).unwrap_or(&u32::MAX);
            if tentative_g < existing {
                came_from.insert(next, pos);
                g_score.insert(next, tentative_g);
                let f = tentative_g + heuristic(next, goal);
                open.push(Node { cost: f, pos: next });
            }
        }
    }
    None
}

fn heuristic(a: IVec2, b: IVec2) -> u32 {
    // Octile distance (for 8-directional movement).
    let dx = (a.x - b.x).unsigned_abs();
    let dy = (a.y - b.y).unsigned_abs();
    10 * (dx.max(dy)) + 4 * (dx.min(dy))
}

/// Generic A* on a navigation graph. `neighbors(pos)` returns `(next_pos, cost)`.
/// `h(pos)` is the admissible heuristic.
pub fn astar_world<P, N, H>(
    start: P,
    goal: P,
    mut neighbors: N,
    mut h: H,
) -> Option<Vec<P>>
where
    P: Copy + Eq + std::hash::Hash + Ord,
    N: FnMut(P) -> Vec<(P, u32)>,
    H: FnMut(P, P) -> u32,
{
    let mut open: BinaryHeap<(std::cmp::Reverse<(u32, P)>,)> = BinaryHeap::new();
    let mut came_from: HashMap<P, P> = HashMap::new();
    let mut g_score: HashMap<P, u32> = HashMap::new();
    g_score.insert(start, 0);
    open.push((std::cmp::Reverse((h(start, goal), start)),));

    while let Some((std::cmp::Reverse((_, pos)),)) = open.pop() {
        if pos == goal {
            let mut path = vec![pos];
            let mut cur = pos;
            while let Some(&prev) = came_from.get(&cur) {
                path.push(prev);
                cur = prev;
            }
            path.reverse();
            return Some(path);
        }
        let cur_g = *g_score.get(&pos).unwrap_or(&u32::MAX);
        for (next, cost) in neighbors(pos) {
            let tentative_g = cur_g.saturating_add(cost);
            let existing = *g_score.get(&next).unwrap_or(&u32::MAX);
            if tentative_g < existing {
                came_from.insert(next, pos);
                g_score.insert(next, tentative_g);
                let f = tentative_g + h(next, goal);
                open.push((std::cmp::Reverse((f, next)),));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astar_straight_line() {
        let mut grid = Grid::new(10, 10);
        // No obstacles — start at (0,0), goal at (5,0).
        let path = astar_grid(&grid, IVec2::new(0, 0), IVec2::new(5, 0)).unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.first().unwrap(), IVec2::new(0, 0));
        assert_eq!(*path.last().unwrap(), IVec2::new(5, 0));
    }

    #[test]
    fn astar_blocked_destination() {
        let mut grid = Grid::new(10, 10);
        grid.set(5, 0, true); // Block the destination.
        let path = astar_grid(&grid, IVec2::new(0, 0), IVec2::new(5, 0));
        assert!(path.is_none());
    }

    #[test]
    fn astar_routes_around_wall() {
        let mut grid = Grid::new(10, 10);
        // Wall straight across row 0 from x=1..4.
        for x in 1..5 {
            grid.set(x, 0, true);
        }
        let path = astar_grid(&grid, IVec2::new(0, 0), IVec2::new(5, 0)).unwrap();
        assert!(!path.is_empty());
        // Path should not pass through any blocked cell.
        for p in &path {
            assert!(!grid.is_blocked(p.x, p.y));
        }
    }
}
