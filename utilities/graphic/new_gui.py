import pygame


# Function to get the color of an agent based on its type and state
def get_agent_color(agent):
    match (
    agent.uid[1], agent.name if hasattr(agent, "name") else None, agent.state if hasattr(agent, "state") else None):
        case (0, _, "active"):  # aep
            return 147, 112, 219
        case (0, _, _):
            return 128, 0, 128

        case (1, "tau", _):  # protein
            return 173, 216, 230
        case (1, _, _):
            return 255, 255, 128

        case (2, "tau", _):  # CleavedProtein
            return 113, 166, 210
        case (2, _, _):
            return 225, 225, 100

        case (3, "tau", _):  # Oligomer
            return 0, 0, 255
        case (3, _, _):
            return 255, 255, 0

        case (4, _, _):  # ExternalInput
            return 169, 169, 169

        case (5, _, _):  # Treatment
            return 211, 211, 211

        case (6, _, "resting"):  # Microglia
            return 144, 238, 144
        case (6, _, _):
            return 0, 100, 0

        case (7, _, "healthy"):  # Neuron
            return 255, 105, 180
        case (7, _, "damaged"):
            return 255, 69, 0
        case (7, _, _):
            return 0, 0, 0

        case (8, _, "pro_inflammatory"):  # Cytokine
            return 255, 0, 0
        case (8, _, _):
            return 0, 255, 255

    return 0, 0, 0  # Default case (fallback color)


class NEWGUI:
    def __init__(self, width, height, gut_context, brain_context, grid_dimensions=(100, 100)):
        self.background_color = (202, 187, 185)
        self.border_color = (255, 255, 255)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.font = pygame.font.Font(None, 36)
        self.running = True
        self.gut_context = gut_context
        self.brain_context = brain_context
        self.grid_width, self.grid_height = grid_dimensions
        self.paused = False
        self.button_rects = []

    # Function to update the screen after each tick
    def update(self, gut_context, brain_context):
        # Update contexts
        self.gut_context, self.brain_context = gut_context, brain_context

        # Fill background and draw border rectangle
        self.screen.fill(self.background_color)

        # Draw border rectangle for good diet context
        inner_rect_good = (50, 40, self.width - 100, self.height - 450)
        pygame.draw.rect(self.screen, self.border_color, inner_rect_good)

        # Draw border rectangle for bad diet context
        inner_rect_bad = (50, self.height - 370, self.width - 100, self.height - 450)
        pygame.draw.rect(self.screen, self.border_color, inner_rect_bad)

        # Draw separating lines
        pygame.draw.line(self.screen, (0, 0, 0), (self.width // 2, inner_rect_good[1]),
                         (self.width // 2, inner_rect_good[1] + inner_rect_good[3]), 4)
        pygame.draw.line(self.screen, (0, 0, 0), (self.width // 2, inner_rect_bad[1]),
                         (self.width // 2, inner_rect_bad[1] + inner_rect_bad[3]), 4)

        # Draw section titles
        text_y_position = inner_rect_good[1] - 30
        self._draw_centered_text("HEALTHY DIET", self.width // 2, text_y_position)
        self._draw_centered_text("Gut Environment", self.width // 4, text_y_position)
        self._draw_centered_text("Brain Environment", 3 * self.width // 4, text_y_position)

        text_y_position = inner_rect_bad[1] - 30
        self._draw_centered_text("UNHEALTHY DIET", self.width // 2, text_y_position)
        self._draw_centered_text("Gut Environment", self.width // 4, text_y_position)
        self._draw_centered_text("Brain Environment", 3 * self.width // 4, text_y_position)

        # Draw buttons and legend
        self.draw_buttons()
        # self.draw_legend()

        # Define areas and draw agents
        healthy_gut_area = (50, 40, self.width // 2 - 47, self.height - 450)
        healthy_brain_area = (self.width // 2 + 3, 40, self.width // 2 - 50, self.height - 450)
        self._draw_agents(self.gut_context.agents(), healthy_gut_area)
        self._draw_agents(self.brain_context.agents(), healthy_brain_area)

        unhealthy_gut_area = (50, self.height - 370, self.width // 2 - 47, self.height - 450)
        unhealthy_brain_area = (self.width // 2 + 3, self.height - 370, self.width // 2 - 50, self.height - 450)
        self._draw_agents(self.gut_context.agents(), unhealthy_gut_area)
        self._draw_agents(self.brain_context.agents(), unhealthy_brain_area)

    def _draw_centered_text(self, text, x_center, y):
        rendered_text = self.font.render(text, True, (0, 0, 0))
        text_x_position = x_center - rendered_text.get_width() // 2
        self.screen.blit(rendered_text, (text_x_position, y))

    # Function to draw the play and pause buttons on the screen
    def draw_buttons(self):
        button_font = pygame.font.Font(None, 20)
        buttons = ["Play", "Pause"]
        button_width = 90
        button_height = 30
        button_spacing = 20

        total_width = (button_width * len(buttons)) + (button_spacing * (len(buttons) - 1))
        start_x = (self.width - total_width) // 2

        for i, button_text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, self.height - 35, button_width, button_height)
            pygame.draw.rect(self.screen, (137, 106, 103), button_rect)
            button_surface = button_font.render(button_text, True, (0, 0, 0))
            # Center the text on the button
            button_text_rect = button_surface.get_rect(center=button_rect.center)
            self.screen.blit(button_surface, button_text_rect.topleft)

            self.button_rects.append((button_rect, button_text))

    def _draw_agents(self, agents, area):
        # pygame.draw.rect(self.screen, area)
        radius = 5

        for agent in agents:
            x_center = area[0] + (agent.pt.x / self.grid_width) * area[2]
            y_center = area[1] + (agent.pt.y / self.grid_height) * area[3]

            # Adjust x and y to keep the entire circle within the area
            x = max(area[0] + radius, min(x_center, area[0] + area[2] - radius))
            y = max(area[1] + radius, min(y_center, area[1] + area[3] - radius))

            color = get_agent_color(agent)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
