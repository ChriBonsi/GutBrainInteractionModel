import pygame

# Class to make the graphics of the simulation
class GUI:
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
        inner_rect = (50, 50, self.width - 100, self.height - 300)
        pygame.draw.rect(self.screen, self.border_color, inner_rect)

        # Draw section titles
        text_y_position = inner_rect[1] - 30
        self._draw_centered_text("Gut Environment", self.width // 4, text_y_position)
        self._draw_centered_text("Brain Environment", 3 * self.width // 4, text_y_position)

        # Draw separating line
        pygame.draw.line(self.screen, (0, 0, 0), (self.width // 2, inner_rect[1]),
                         (self.width // 2, inner_rect[1] + inner_rect[3]), 4)

        # Draw buttons and legend
        self.draw_buttons()
        self.draw_legend()

        # Define areas and draw agents
        gut_area = (50, 50, self.width // 2 - 47, self.height - 300)
        brain_area = (self.width // 2 + 3, 50, self.width // 2 - 50, self.height - 300)
        self._draw_context_agents(self.gut_context, gut_area)
        self._draw_context_agents(self.brain_context, brain_area)

    def _draw_centered_text(self, text, x_center, y):
        rendered_text = self.font.render(text, True, (0, 0, 0))
        text_x_position = x_center - rendered_text.get_width() // 2
        self.screen.blit(rendered_text, (text_x_position, y))

    def _draw_context_agents(self, context, area):
        self.draw_agents(context.agents(), area)

    # Function to draw agents on the screen
    def draw_agents(self, agents, area):
        radius = 5

        for agent in agents:
            x_center = area[0] + (agent.pt.x / self.grid_width) * area[2]
            y_center = area[1] + (agent.pt.y / self.grid_height) * area[3]

            # Adjust x and y to keep the entire circle within the area
            x = max(area[0] + radius, min(x_center, area[0] + area[2] - radius))
            y = max(area[1] + radius, min(y_center, area[1] + area[3] - radius))

            color = self.get_agent_color(agent)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    # Function to get the color of an agent based on its type and state
    def get_agent_color(self, agent):
        if agent.uid[1] == 0: #aep
            if agent.state == "active":
                color = (147, 112, 219)
            else:
                color = (128, 0, 128)
        elif agent.uid[1] == 1: #protein
            if agent.name == "tau":
                color = (173, 216, 230)  # Light Blue
            else:
                color = (255, 255, 128)  # Light Yellow 
        elif agent.uid[1] == 2: #CleavedProtein
            if agent.name == "tau":
                color = (113, 166, 210)  # Darker Blue               
            else:
                color = (225, 225, 100)  # Darker Yellow
        elif agent.uid[1] == 3: #Oligomer
            if agent.name == "tau":
                color = (0, 0, 255)  # Blue
            else:
                color = (255, 255, 0)  # Yellow
        elif agent.uid[1] == 4: #ExternalInput
            color = (169, 169, 169)  # Dark Grey
        elif agent.uid[1] == 5: #Treatment
            color = (211, 211, 211)  # Light Grey
        elif agent.uid[1] == 6: #Microglia
            if agent.state == "resting":
                color = (144, 238, 144)  # Light Green 
            else:
                color = (0, 100, 0)  # Dark Green
        elif agent.uid[1] == 7: #Neuron
            if agent.state == "healthy":
                color = (255, 105, 180)  # Pink
            elif agent.state == "damaged":
                color = (255, 69, 0)  # Orange-Red
            else:
                color = (0, 0, 0)  # Black
        elif agent.uid[1] == 8: #Cytokine
            if agent.state == "pro_inflammatory":
                color = (255, 0, 0)  # Red
            else:
                color = (0, 255, 255)  # Cyan
        return color

    # Function to draw the legend on the screen
    def draw_legend(self):
        # Define legend position and size
        legend_x = 60
        legend_y = self.height - 250
        legend_radius = 6
        legend_spacing = 35
        text_offset_x = 15
        row_spacing = 20

        # Define legend title
        legend_title_font = pygame.font.Font(None, 20)
        legend_title_surface = legend_title_font.render("Legend:", True, (0, 0, 0))
        title_width, title_height = legend_title_surface.get_size()

        # Define legend fonts
        legend_font = pygame.font.Font(None, 18)
        legend_color = (0, 0, 0)  # Black

        # Define legend items
        legend_items = {(147, 112, 219): "Active AEP", (128, 0, 128): "Hyperactive AEP", (173, 216, 230): "Tau Protein",
                        (255, 255, 128): "Alpha-syn Protein", (113, 166, 210): "Tau Cleaved",
                        (225, 225, 100): "Alpha-syn Cleaved", (0, 0, 255): "Tau Oligomer",
                        (255, 255, 0): "Alpha-syn Oligomer", (169, 169, 169): "External Input",
                        (211, 211, 211): "Treatment", (144, 238, 144): "Resting Microglia",
                        (0, 100, 0): "Active Microglia", (255, 105, 180): "Healthy Neuron",
                        (255, 69, 0): "Damaged Neuron", (0, 0, 0): "Dead Neuron",
                        (255, 0, 0): "Pro-inflammatory Cytokine", (0, 255, 255): "Anti-inflammatory Cytokine"}

        # Calculate number of items per column
        items_per_column = len(legend_items) // 2 + len(legend_items) % 2

        # Calculate legend background size
        legend_width = 400
        legend_height = items_per_column * row_spacing + 70

        # Calculate legend title position
        title_x = legend_x + (legend_width - title_width) // 2
        title_y = legend_y + 10

        # Draw legend title
        self.screen.blit(legend_title_surface, (title_x, title_y))

        # Draw legend items
        row = 0
        for color, label in legend_items.items():
            col = 0 if row < items_per_column else 1
            circle_x = legend_x + col * (legend_width // 2) + legend_radius
            circle_y = legend_y + 35 + (row % items_per_column) * row_spacing + legend_radius
            pygame.draw.circle(self.screen, color, (circle_x, circle_y), legend_radius)
            legend_text_surface = legend_font.render(label, True, legend_color)
            text_width, text_height = legend_text_surface.get_size()
            text_x = legend_x + col * (legend_width // 2) + 2 * legend_radius + text_offset_x
            text_y = circle_y - text_height // 2
            self.screen.blit(legend_text_surface, (text_x, text_y))
            row += 1

    # Function to draw the play and stop buttons on the screen
    def draw_buttons(self):
        button_font = pygame.font.Font(None, 30)
        buttons = ["Play", "Stop"]
        button_width = 120
        button_height = 40
        button_spacing = 10

        total_width = (button_width * len(buttons)) + (button_spacing * (len(buttons) - 1))
        start_x = (self.width - total_width) // 2

        for i, button_text in enumerate(buttons):
            button_x = start_x + i * (button_width + button_spacing)
            button_rect = pygame.Rect(button_x, self.height - 150, button_width, button_height)
            pygame.draw.rect(self.screen, (137, 106, 103), button_rect)
            button_surface = button_font.render(button_text, True, (0, 0, 0))
            # Center the text on the button
            button_text_rect = button_surface.get_rect(center=button_rect.center)
            self.screen.blit(button_surface, button_text_rect.topleft)

            self.button_rects.append((button_rect, button_text))

    # Function to handle mouse clicks on the buttons
    def handle_button_click(self, mouse_pos):
        for button_rect, button_text in self.button_rects:
            if button_rect.collidepoint(mouse_pos):
                self.on_button_click(button_text)

    # Function to handle button clicks
    def on_button_click(self, button_text):
        if button_text == "Play":
            self.paused = False
        elif button_text == "Stop":
            self.paused = True