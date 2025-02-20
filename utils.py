import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider
import numpy as np


def find_events_custom(trigger_signal, min_threshold, max_threshold, min_duration):
    """
    Detect events in a 1D trigger signal.

    Parameters:
    - trigger_signal (numpy array): The 1D array of trigger data.
    - min_threshold (float): The minimum amplitude threshold to detect an event.
    - max_threshold (float): The maximum amplitude threshold to filter events.
    - min_duration (int): Minimum number of consecutive samples to count as an event.

    Returns:
    - events (numpy array): An Nx2 array where each row contains [onset, offset] of an event.
    """
    above_thresh = (trigger_signal >= min_threshold) & (trigger_signal <= max_threshold)  # Boolean array of valid events
    
    # Find transitions
    diff_signal = np.diff(np.concatenate(([0], above_thresh.astype(int), [0])))
    onsets = np.where(diff_signal == 1)[0]  # Rising edges
    offsets = np.where(diff_signal == -1)[0]  # Falling edges
    
    # Ensure each event is a continuous block before returning to baseline
    valid_events = []
    for onset, offset in zip(onsets, offsets):
        if (offset - onset) >= min_duration:
            valid_events.append((onset, offset))
    
    return np.array(valid_events) if valid_events else np.empty((0, 2), dtype=int)



def plot_trigger_data_and_events(trig_data, events):
    fig, ax = plt.subplots(figsize=[20,5])
    ax.plot(trig_data)
    for ee in events:
        ax.axvline(ee[0], color='red')
    ax.set_title('N_events=%s'%len(events))



class EventViewer:
    def __init__(self, trigger_data, events, initial_zoom_width=50000):
        self.trigger_data = trigger_data
        self.events = list(events.copy())
        self.lines = []
        self.zoom_width = initial_zoom_width
        self._saved = False  # Flag to track if save was clicked
        
        # Create the main figure and axes
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
        plt.subplots_adjust(bottom=0.2)
        
        # Initialize the plots
        self.setup_plots()
        self.setup_controls()
        
        # Store the current view limits
        self.current_xlim = self.ax1.get_xlim()
        self.current_ylim = self.ax1.get_ylim()

        
    def setup_plots(self):
        # Main plot
        self.ax1.plot(self.trigger_data, 'b-', label='Trigger Signal')
        self.update_event_lines()
        self.ax1.set_title('Event Viewer - Click to add/remove events')
        
        # Overview plot
        self.ax2.plot(self.trigger_data, 'b-', alpha=0.5)
        self.update_overview_lines()
        self.ax2.set_title('Overview - Drag in this plot to zoom main plot')
        
    def update_event_lines(self):
        # Clear existing lines
        for line in self.lines:
            line.remove()
        self.lines = []
        
        # Add new lines
        for ee in self.events:
            line = self.ax1.axvline(ee[0], color='red', alpha=0.5)
            self.lines.append(line)
            
    def update_overview_lines(self):
        self.ax2.clear()
        self.ax2.plot(self.trigger_data, 'b-', alpha=0.5)
        for ee in self.events:
            self.ax2.axvline(ee[0], color='red', alpha=0.2)
        self.ax2.set_title('Overview - Drag in this plot to zoom main plot')
        
    def setup_controls(self):
        # Add buttons
        self.button_ax_save = plt.axes([0.15, 0.05, 0.1, 0.04])
        self.button_ax_reset = plt.axes([0.3, 0.05, 0.1, 0.04])
        self.button_ax_help = plt.axes([0.45, 0.05, 0.1, 0.04])
        
        self.button_save = Button(self.button_ax_save, 'Save')
        self.button_reset = Button(self.button_ax_reset, 'Reset')
        self.button_help = Button(self.button_ax_help, 'Help')
        
        # Add goto textbox
        self.textbox_ax = plt.axes([0.6, 0.05, 0.15, 0.04])
        self.textbox = TextBox(self.textbox_ax, 'Goto Event:', initial='0')
        
        # Add zoom width slider
        self.slider_ax = plt.axes([0.15, 0.1, 0.6, 0.02])
        self.zoom_slider = Slider(
            self.slider_ax, 'Zoom Width', 
            valmin=1000,  # Minimum zoom width
            valmax=200000,  # Maximum zoom width
            valinit=self.zoom_width,
            valstep=1000  # Step size
        )
        
        # Add zoom width textbox for precise control
        self.zoom_textbox_ax = plt.axes([0.8, 0.1, 0.15, 0.02])
        self.zoom_textbox = TextBox(
            self.zoom_textbox_ax, 
            'Width:', 
            initial=str(self.zoom_width)
        )
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.button_save.on_clicked(self.save_events)
        self.button_reset.on_clicked(self.reset_view)
        self.button_help.on_clicked(self.show_help)
        self.textbox.on_submit(self.goto_event)
        self.zoom_slider.on_changed(self.update_zoom_width)
        self.zoom_textbox.on_submit(self.update_zoom_width_from_text)
        
        # Display initial help message
        self.show_help(None)
    
    def update_zoom_width_from_text(self, text):
        try:
            new_width = int(text)
            if 1000 <= new_width <= 200000:
                self.zoom_width = new_width
                self.zoom_slider.set_val(new_width)
                # Update current view while maintaining center
                center = np.mean(self.ax1.get_xlim())
                self.ax1.set_xlim(center - self.zoom_width/2, center + self.zoom_width/2)
                self.fig.canvas.draw_idle()
        except ValueError:
            pass
    
    def update_zoom_width(self, val):
        self.zoom_width = int(val)
        self.zoom_textbox.set_val(str(self.zoom_width))
        # Update current view while maintaining center
        center = np.mean(self.ax1.get_xlim())
        self.ax1.set_xlim(center - self.zoom_width/2, center + self.zoom_width/2)
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        if event.inaxes == self.ax2:
            # Clicking in overview plot sets the view in main plot
            self.ax1.set_xlim(event.xdata - self.zoom_width/2, event.xdata + self.zoom_width/2)
            self.fig.canvas.draw_idle()
        elif event.inaxes == self.ax1:
            # Find closest event to click
            click_x = event.xdata
            if self.events:
                distances = [abs(ee[0] - click_x) for ee in self.events]
                closest_idx = np.argmin(distances)
                min_distance = distances[closest_idx]
                
                if min_distance < 1000:  # If click is close to an event, remove it
                    self.events.pop(closest_idx)
                else:  # If click is not near any event, add new one
                    new_event = [int(click_x), int(click_x) + 100]
                    self.events.append(new_event)
                    # Sort events by onset time
                    self.events.sort(key=lambda x: x[0])
            else:  # If no events exist, add the first one
                new_event = [int(click_x), int(click_x) + 100]
                self.events.append(new_event)
            
            # Update both plots
            self.update_event_lines()
            self.update_overview_lines()
            self.fig.canvas.draw_idle()
    
    def goto_event(self, text):
        try:
            event_idx = int(text)
            if 0 <= event_idx < len(self.events):
                event_x = self.events[event_idx][0]
                self.ax1.set_xlim(event_x - self.zoom_width/2, event_x + self.zoom_width/2)
                self.fig.canvas.draw_idle()
        except ValueError:
            pass
    
    def reset_view(self, event):
        self.ax1.set_xlim(0, len(self.trigger_data))
        self.ax1.set_ylim(min(self.trigger_data), max(self.trigger_data))
        self.fig.canvas.draw_idle()
    
    def show_help(self, event):
        help_text = """
        Instructions:
        - Click anywhere to add a new event
        - Click on existing event to remove it
        - Click in bottom plot to zoom to that area in top plot
        - Use slider or text box to adjust zoom width
        - Use 'Goto Event' to jump to specific event number
        - 'Save' will return current events
        - 'Reset' will reset the view
        """
        self.ax1.set_title(help_text)
        self.fig.canvas.draw_idle()
    
    def save_events(self, event=None):
        """Called when save button is clicked"""
        self._saved = True
        plt.close(self.fig)
    
    def get_kept_events(self):
        return self.saved_events if self.saved_events is not None else np.array(self.events)

def manual_event_filter(trigger_data, events, initial_zoom_width=50000):
    """
    Launch interactive viewer to manually filter events.
    
    Parameters:
    - trigger_data: numpy array of the trigger signal
    - events: numpy array of event times (Nx2 array)
    - initial_zoom_width: initial width of the zoom window (default: 50000)
    
    Returns:
    - numpy array of kept events
    """
    # Create viewer instance
    viewer = EventViewer(trigger_data, events, initial_zoom_width)
    
    # Show the plot and wait for it to close
    plt.show(block=True)
    
    # Convert events to numpy array before returning
    if viewer._saved:
        final_events = np.array(viewer.events)
    else:
        final_events = events  # Return original events if save wasn't clicked
        
    # Clean up
    plt.close('all')
    
    return final_events
