import pygame
import torch
import math
import os
import sys

from HeXO import HeXOGame
from model import GNNModel
from mcts import MCTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# GUI Constants
HEX_SIZE = 25
WIDTH = 800
HEIGHT = 600
FPS = 30
BG_COLOR = (30, 30, 30)
GRID_COLOR = (80, 80, 80)
HIGHLIGHT_COLOR = (120, 120, 120)
TEXT_COLOR = (220, 220, 220)

P1_COLOR = (240, 80, 80)   # Red
P2_COLOR = (80, 180, 240)  # Blue

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HeXO - AI GUI")
font = pygame.font.SysFont("Arial", 20)
large_font = pygame.font.SysFont("Arial", 40)

def load_model():
    model = GNNModel().to(DEVICE)
    model_path = "model_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        print(f"Loaded {model_path}")
    else:
        print("Using un-trained random model.")
    model.eval()
    return model

def axial_to_pixel(q, r, cx, cy):
    x = HEX_SIZE * math.sqrt(3) * (q + r / 2.0)
    y = HEX_SIZE * 3.0 / 2.0 * r
    return cx + x, cy + y

def pixel_to_axial(px, py, cx, cy):
    x = px - cx
    y = py - cy
    q = (math.sqrt(3)/3 * x - 1.0/3 * y) / HEX_SIZE
    r = (2.0/3 * y) / HEX_SIZE
    
    rx = round(q)
    ry = round(-q - r)
    rz = round(r)
    
    x_diff = abs(rx - q)
    y_diff = abs(ry - (-q - r))
    z_diff = abs(rz - r)
    
    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry
        
    return int(rx), int(rz)

def get_hex_corners(hx, hy):
    corners = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = math.pi / 180 * angle_deg
        corners.append((hx + HEX_SIZE * math.cos(angle_rad),
                        hy + HEX_SIZE * math.sin(angle_rad)))
    return corners

def get_ai_move(model, game, num_sims=50):
    mcts_engine = MCTS(model)
    root = mcts_engine.search(game, num_sims)
    best_move, _ = mcts_engine.get_policy(root, temperature=0)
    return best_move

def main():
    model = load_model()
    game = HeXOGame()
    
    clock = pygame.time.Clock()
    running = True
    
    human_player = 1
    ai_player = 2
    
    # Camera
    cx, cy = WIDTH // 2, HEIGHT // 2
    
    hover_hex = None
    error_msg = ""
    
    while running:
        legal_moves = game.get_legal_moves()
            
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left click -> Place piece
                if not game.done and game.current_player == human_player:
                    if hover_hex in legal_moves:
                        game.step(*hover_hex)
                        error_msg = ""
                    else:
                        error_msg = "Invalid move (Too far or cell occupied)"
                        
            elif event.type == pygame.MOUSEMOTION:
                # Update hover
                mx, my = event.pos
                hover_hex = pixel_to_axial(mx, my, cx, cy)
                
        # Handle AI Turn (Blocking)
        if not game.done and game.current_player == ai_player:
            # Draw one frame to say "Thinking..."
            screen.fill(BG_COLOR)
            thinking_text = font.render(f"AI is thinking...", True, (255, 255, 100))
            screen.blit(thinking_text, (20, HEIGHT - 40))
            pygame.display.flip()
            
            # Flush events to prevent OS freeze warnings
            pygame.event.pump()
            
            best_move = get_ai_move(model, game, num_sims=50) # Reduced sims for faster GUI response
            game.step(*best_move)
            
            # Clear events so human doesn't accidentally click during AI delay
            pygame.event.clear()

        # Render
        screen.fill(BG_COLOR)
        
        draw_hexes = set(game.board.keys()).union(legal_moves)
        
        for (q, r) in draw_hexes:
            hx, hy = axial_to_pixel(q, r, cx, cy)
            # Only draw visible hexes
            if -HEX_SIZE < hx < WIDTH+HEX_SIZE and -HEX_SIZE < hy < HEIGHT+HEX_SIZE:
                corners = get_hex_corners(hx, hy)
                
                if (q, r) in game.board:
                    # Occupied
                    pygame.draw.polygon(screen, GRID_COLOR, corners, 1)
                    owner = game.board[(q, r)]
                    color = P1_COLOR if owner == 1 else P2_COLOR
                    pygame.draw.circle(screen, color, (int(hx), int(hy)), int(HEX_SIZE * 0.7))
                else:
                    # Empty but legal
                    if hover_hex == (q, r):
                        pygame.draw.polygon(screen, HIGHLIGHT_COLOR, corners)
                    pygame.draw.polygon(screen, GRID_COLOR, corners, 1)
                    
        # UI Text
        expected = 1 if (game.turn_number == 1 and game.current_player == 1) else 2
        remaining = expected - game.placements_this_turn
        
        p1_pieces = sum(1 for v in game.board.values() if v == 1)
        p2_pieces = sum(1 for v in game.board.values() if v == 2)
        
        info_str = f"P1 (Red): {p1_pieces}   |   P2 AI (Blue): {p2_pieces}"
        screen.blit(font.render(info_str, True, TEXT_COLOR), (20, 20))
        
        if not game.done:
            turn_str = f"Current Turn: Player {game.current_player}  |  Pieces needed: {remaining}"
            screen.blit(font.render(turn_str, True, TEXT_COLOR), (20, 50))
            
            controls_str = "L-Click: Place"
            screen.blit(font.render(controls_str, True, (150, 150, 150)), (WIDTH - 150, 20))
            
            if error_msg:
                screen.blit(font.render(error_msg, True, (255, 100, 100)), (20, 80))
        else:
            win_str = f"Player {game.winner} Wins!" if game.winner else "Draw!"
            win_surf = large_font.render(win_str, True, (255, 215, 0))
            screen.blit(win_surf, (WIDTH//2 - win_surf.get_width()//2, HEIGHT - 100))
            
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
